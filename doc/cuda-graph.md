# CUDA graph decode

## Scope

`PagedDecodeGraph` captures one complete numerical decode step for a paged
cache shape:

```text
token embedding -> decoder layers -> paged attention -> final norm -> LM head
```

The captured graph includes the projection kernels used by the loaded model
and vLLM's variable-length FlashAttention over the paged KV cache. Token
sampling, request iteration, input validation, cache allocation, and host-side
cache bookkeeping remain outside the graph.

CUDA graph replay is enabled by default when all of these conditions hold:

- the engine has `use_cuda_graph=True`;
- the model and cache are on CUDA;
- the request uses `SequenceCacheHandle`, the paged-cache implementation; and
- the model derives from `CausalLMBase`, whose decoder and LM-head operations
  are captured directly.

Dense caches, CPU, MPS, and alternate runtime-model implementations retain the
ordinary eager decode path. Set `use_cuda_graph=False` on `Engine` or pass it
to `Engine.from_model_dir()` to force eager decode while still using the paged
cache.

## Capture and replay

### Graph capture

Generation first prefills the request cache normally. A graph is created
lazily only if generation needs a decode after the first sampled token. This
avoids capture for requests that finish immediately because of EOS, the token
limit, or the context limit.

Capture allocates persistent CUDA tensors for:

- one `[1, 1]` `torch.long` input token;
- one `[1, 1]` `torch.long` position ID;
- one block table sized for the model's maximum sequence length; and
- the output logits owned by the graph.

The graph records `DecoderModel.forward()` followed by the LM-head projection.

### PagedGraphCache

The captured graph needs a fixed-size block table and maximum cache capacity,
while a request cache must describe only the blocks and capacity actually
allocated to that request. Using the request cache for both roles would require
temporarily replacing its allocation metadata during capture.

`PagedGraphCache` keeps those roles separate. Its capture-facing layer views
use the persistent block table, maximum sequence capacity, and the same
physical K/V pool as the request. Its host-facing state delegates length,
capacity checks, and rollback to the currently bound request cache.

CUDA replay uses the captured table and pool addresses directly. The host still
needs the bound request length to update the position tensor and keep logical
cache state synchronized with the K/V writes. Binding another request copies
its allocated block IDs into the persistent table prefix and changes which
request receives that host-side bookkeeping.

### Replay

The request's first replay overwrites the placeholder K/V data written during
capture.

The table shape does not depend on request capacity or physical pool size, so
requests with different cache capacities can share the graph. Replacing the
pool replaces the retained graph because capture fixes the K/V storage
addresses.

For every replay, the host performs four small operations:

1. copy the next token into the persistent input tensor;
2. fill the persistent position tensor from the current cache length;
3. advance the host-visible cache length by one; and
4. call `CUDAGraph.replay()`.

The device position tensor makes one captured graph valid across changing
logical sequence lengths. Each layer uses it both to select the physical cache
slot for the new K/V values and to pass the effective key length to variable-
length FlashAttention. The graph therefore keeps fixed tensor shapes and cache
capacity while attention observes only the valid logical prefix.

Python assignments made by `PagedLayerKVCache.write()` run during capture but
do not run again during replay. `SequenceCacheHandle.advance()` is consequently
required to keep the host length synchronized with the device writes. If
replay raises immediately, the runtime restores the previous host length.

No explicit model warm-up is required before capture. Zero-warm-up correctness
tests pass for the small dense model and the real GPTQ INT4 and INT8 Qwen
checkpoints.

The graph and its persistent tensors are engine-owned, while request caches are
released on completion, stream closure, or failure. Returned logits are
graph-owned storage and will be overwritten by the next replay; generation
consumes them before that happens. Mutable graph inputs assume the runtime's
current single-active-request execution.

## Decode benchmark

The benchmark isolates model decode from sampling and text processing so it
measures the kernel-launch reduction directly.

- GPU: NVIDIA GeForce RTX 3070, 8 GiB
- NVIDIA driver: 580.173.02
- CUDA: 13.0
- PyTorch: 2.13.0+cu130
- Models: Qwen3-0.6B FP16, GPTQ INT8, and GPTQ INT4
- Cache: paged, with variable-length FlashAttention in both paths
- Prompt: 32 synthetic tokens
- Decode: 64 fixed tokens per measured run
- Warm-up: one unmeasured eight-token request per mode
- Repetitions: 3
- Reported values: medians with one device synchronization around each run

Graph construction was measured separately after prefill. The decode timing
includes the Python replay calls and input copies, but excludes graph capture,
prefill, token sampling, and tokenizer work.

| Weights | Eager TPOT | Eager tokens/s | Graph TPOT | Graph tokens/s | Speedup | Capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 20.3534 ms | 49.13 | 5.8647 ms | 170.51 | 3.470x | 20.50 ms |
| GPTQ INT8 | 22.1754 ms | 45.09 | 4.7999 ms | 208.34 | 4.620x | 22.38 ms |
| GPTQ INT4 | 22.0685 ms | 45.31 | 4.2724 ms | 234.06 | 5.165x | 22.41 ms |

Eager GPTQ decode was slightly slower than FP16 because its additional Marlin
dispatch work was paid for every layer and token. Capturing the complete decode
removes most repeated CPU launch overhead: INT8 becomes 22% faster than the
captured FP16 model, and INT4 becomes 37% faster. The remaining graph replay
time then exposes the lower-cost quantized matrix multiplications.

At these measurements, the roughly 20-22 ms initial capture cost is
recovered after two decoded tokens. This crossover excludes sampling overhead
and is specific to the tested hardware, software versions, model, batch size,
and sequence lengths. End-to-end serving results can differ because sampling
and other host work are not captured.

Four sequential requests with the same six-block cache shape measured the
first cold capture separately from later block-table rebinds:

| Weights | First capture | Median rebind | Reused decode tokens/s |
| --- | ---: | ---: | ---: |
| FP16 | 86.388 ms | 0.032 ms | 171.26 |
| GPTQ INT8 | 22.313 ms | 0.028 ms | 209.80 |
| GPTQ INT4 | 22.056 ms | 0.030 ms | 232.14 |

For a 32-token prompt, the fixed 2,560-entry table required by Qwen3's 40,960-
token context measured 5.8906 ms per decoded token, versus 5.7695 ms with a
request-sized table. This 2.1% difference is small relative to the recaptures
avoided across request lengths. Unused table entries do not allocate KV blocks.
