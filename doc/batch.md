# Batched decode

## Implementations

The portable batch path combines independent request caches into zero-padded
contiguous tensors and calls PyTorch SDPA. It is used for CPU, MPS, and dense
caches. For a paged cache, this reference path gathers every request's physical
blocks before constructing the padded batch and materializes repeated K/V heads
for grouped-query attention.

The CUDA paged path keeps each request history in the shared physical block
pool. It flattens the query tokens and passes request boundaries, effective K/V
lengths, and a batched block table to variable-length FlashAttention. Only the
block-table rows are padded; the kernel does not gather or pad the actual K/V
states and handles grouped-query attention without materializing repeated K/V
heads.

Decode contributes one query token per active request, so its query tensor is
naturally rectangular. The ragged dimension is the cached K/V history: each
request can have a different logical length and block count. Ragged batched
prefill is not implemented; prompts are still prefetched sequentially.

## Long-context benchmark

All decode variants were measured with the following setup:

- GPU: NVIDIA GeForce RTX 3070, 8 GiB
- Model: Qwen3-0.6B FP16
- Prompt: 509 actual tokens per request
- Generation: 128 tokens per request
- Cache: paged FP16, 16-token blocks
- Warm-up: one complete unmeasured generation
- Repetitions: 3
- Reported values: medians

![Padded SDPA, eager paged, and CUDA-graph paged decode performance](../img/batch-decode-performance.png)

`Decode step` is the latency to produce one token for every active request. It
is also the user-visible inter-token latency for each request in a fixed batch.
`Decode tokens/s` is aggregate batch throughput. Prefill and sampling are
excluded from these decode measurements.

| Batch | Padded SDPA step | Padded SDPA tok/s | Ragged paged step | Ragged paged tok/s | Throughput gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 26.22 ms | 76.25 | 23.39 ms | 85.52 | 12.2% |
| 8 | 45.68 ms | 175.22 | 25.08 ms | 318.96 | 82.0% |
| 16 | 69.12 ms | 231.63 | 27.69 ms | 577.82 | 149.5% |
| 24 | 92.16 ms | 260.39 | 29.52 ms | 812.99 | 212.2% |
| 32 | 115.20 ms | 277.70 | 32.04 ms | 998.68 | 259.6% |
| 40 | 140.80 ms | 284.39 | 34.45 ms | 1161.04 | 308.3% |
| 48 | 163.20 ms | 294.40 | 35.82 ms | 1339.92 | 355.1% |
| 64 | 211.84 ms | 301.68 | 41.82 ms | 1530.38 | 407.3% |

CUDA graph replay captures the complete dense FP16 decode path, including
paged K/V writes and variable-length FlashAttention. It removes most eager
kernel-launch overhead:

| Batch | Eager paged step | Graph paged step | Eager paged tok/s | Graph paged tok/s | Graph speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 23.39 ms | 6.56 ms | 85.52 | 304.86 | 3.56x |
| 8 | 25.08 ms | 7.85 ms | 318.96 | 1019.71 | 3.20x |
| 16 | 27.69 ms | 9.49 ms | 577.82 | 1686.40 | 2.92x |
| 32 | 32.04 ms | 12.64 ms | 998.68 | 2531.78 | 2.54x |
| 64 | 41.82 ms | 18.29 ms | 1530.38 | 3499.26 | 2.29x |

These are steady-state replay measurements; one-time graph capture is excluded.
The graph is recaptured when early termination changes the active batch size.

Additional eager ragged-paged measurements probe its practical saturation point:

| Batch | Decode step | Decode tok/s | Reserved request cache |
| ---: | ---: | ---: | ---: |
| 56 | 38.27 ms | 1463.30 | 3.83 GiB |
| 64 | 41.82 ms | 1530.38 | 4.38 GiB |
| 72 | 44.11 ms | 1632.14 | 4.92 GiB |

Batch 64 to 72 increases aggregate throughput by another 6.6%. The measured
range therefore reaches the GPU memory limit before showing a clear throughput
plateau: batch 80 could not allocate its fixed KV pool on the 8 GiB GPU.

## Interpretation

The improvement grows with batch size because the padded path creates
increasing amounts of redundant K/V traffic. It gathers pages, constructs a
contiguous padded batch, and expands GQA K/V heads before SDPA. The ragged path
reads the original cache through each request's block table and logical length.
It also writes the new K/V states for the whole batch with one operation per
tensor and layer instead of one operation per request. Together these avoid the
dominant request-scaled copies and reach 407.3% higher throughput at batch 64.

Paged attention can show lower sampled GPU utilization while achieving higher
throughput. Allocated VRAM and GPU utilization measure different things: the
fixed KV pool keeps VRAM occupied for its lifetime, whereas utilization reports
how often kernels execute during the sampling interval. Removing copies makes
kernels finish sooner and exposes eager launch gaps, block-table construction,
and host-side sampling.

The benchmark's cache column reports active request reservations, not the
physical pool allocation. With `max_seq_len=704`, the batch-64 physical pool is
4.81 GiB while active requests reserve 4.38 GiB (640 rounded positions each).
The remaining 448 MiB is unused pool capacity. Model weights, CUDA state,
activations, attention workspace, and allocator-retained buffers consume the
rest of VRAM.

CUDA graph replay then removes most of the remaining launch overhead. Its
relative speedup falls from 3.56x at batch 2 to 2.29x at batch 64 as GPU compute
and memory traffic become a larger fraction of each step. Caching graphs by
batch size or using fixed-size buckets remains future work for avoiding
recapture when requests terminate.
