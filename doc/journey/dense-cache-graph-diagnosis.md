# Dense-cache CUDA graph failure diagnosis

## Summary

Repeated full-model CUDA graph capture with a dense KV cache produced
non-finite GPTQ INT8 logits. The graph lifecycle and Marlin kernels were not
the cause. Dense cache storage was allocated with `torch.empty`, while capture
expanded attention to a fixed 16-token bucket. SDPA could therefore consume
non-finite values from masked but unwritten cache positions before applying
the attention mask.

Dense cache allocation now initializes K/V storage to zero. This establishes
the invariant that every unwritten position is finite and makes repeated
dense-cache graph capture deterministic.

## Symptom

The first graph usually matched eager decode, but a later graph captured on a
new dense cache could return all-NaN logits. The failure depended on allocator
history: newly allocated storage sometimes reused memory containing non-finite
values.

The same model remained correct with paged-cache graph replay. This difference
was important because paged variable-length FlashAttention receives the exact
logical key length through `seqused_k`, whereas dense SDPA capture receives a
bucket-sized K/V view plus a mask.

## Investigation

The experiments used Qwen3-0.6B GPTQ INT8 on an RTX 3070 with CUDA 13.0 and
PyTorch 2.13.0+cu130.

| Experiment | Result |
| --- | --- |
| Keep the previous graph alive | Failure remained |
| Destroy the previous graph before recapture | Failure remained |
| Clear every active Marlin workspace | Failure remained; workspaces were already zero |
| Force SDPA math backend | Failure remained |
| Force SDPA memory-efficient backend | Failure remained |
| Capture repeatedly without changing graph ownership | Capture-time output stayed finite; replay could fail |
| Repeat paged INT8 capture for three requests | Every request matched eager |
| Trace quantized dense graph intermediates | Layer 1 QKV was finite; the SDPA result was the first non-finite tensor |
| Inspect the new cache's unwritten bucket padding | Layer 1 contained 333 non-finite values |
| Zero dense K/V storage before prefill | Three repeated captures all matched eager |

These results rule out stale Marlin semaphore state, graph destruction, and a
specific fused SDPA backend. They localize the failure to SDPA reading
uninitialized dense-cache padding.

## Root cause

Ordinary eager dense attention returns the exact valid K/V prefix. CUDA graph
capture cannot change tensor shapes on every replay, so `DenseLayerKVCache`
rounds its attention view to the next 16-token boundary during capture. For
example, writing position 32 produces a view covering positions 0 through 47,
although only positions 0 through 32 are valid.

The boolean attention mask logically excludes positions 33 through 47, but a
fused attention implementation is allowed to load those K/V elements and form
intermediate products before applying the mask. IEEE arithmetic makes this
unsafe for uninitialized data:

```text
finite query * NaN key = NaN
NaN + masked bias = NaN
```

Masking controls attention semantics; it does not sanitize non-finite tensor
contents. `torch.empty` consequently violated the finite-padding invariant
required by bucketed dense attention.

## Fix

`DenseKVCache` now uses `torch.zeros` for key and value allocation. Prefill and
decode overwrite valid positions as before, while unwritten positions remain
finite. Resetting a cache can retain old K/V values safely because model output
is finite and the mask still excludes positions beyond the new logical length.

Zero-initializing the full cache is deliberately simpler than adding a graph-
specific padding API or recording zeroing operations inside the graph. A
captured zero of a fixed slice could erase positions that become valid on later
replays. If allocation initialization becomes measurable at large capacities,
a future dense graph runner may zero only the next bucket outside capture, but
that optimization should preserve the same finite-padding invariant.

## Corrected benchmark

The corrected experiment used a 32-token prompt, 64 fixed decode tokens, four
16-token dense attention buckets, and the median of three measured runs.
Graph-only throughput excludes capture; effective throughput includes all four
request-local captures.

| Weights | Eager | Graph replay | Replay speedup | Four captures | Effective throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 51.80 tok/s | 165.64 tok/s | 3.20x | 77.12 ms | 138.08 tok/s |
| GPTQ INT8 | 47.00 tok/s | 201.09 tok/s | 4.28x | 87.89 ms | 157.57 tok/s |
| GPTQ INT4 | 47.13 tok/s | 225.24 tok/s | 4.78x | 87.79 ms | 172.07 tok/s |

All corrected graph outputs matched eager decode. Dense graph replay is fast,
but it needs a new graph at every attention-length bucket. The existing paged
path remains preferable for decode because one variable-length FlashAttention
graph covers the request's full capacity.
