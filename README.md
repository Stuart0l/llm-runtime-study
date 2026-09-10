# Mini LLM Inference Runtime

A study-oriented text-generation runtime implemented directly with PyTorch.
It loads local Safetensors checkpoints without Transformers, runs on CPU,
NVIDIA CUDA, or Apple MPS, and exposes both a command-line interface and a
synchronous OpenAI-compatible Chat Completions endpoint.

The runtime currently supports:

| Area | Support |
| --- | --- |
| Architectures | Qwen3 dense and IBM Granite 3.1 sparse MoE |
| Checkpoints | Single-file and indexed sharded Safetensors |
| Tokenization | Local `tokenizer.json` through `tokenizers` |
| Generation | Greedy, temperature, top-k, top-p, and seeded sampling |
| Execution | CPU, NVIDIA CUDA, and Apple MPS; batch size one; one active request |
| Cache | Runtime-owned request caches; paged by default, dense optional |
| Serving | Synchronous OpenAI-compatible Chat Completions through FastAPI |

### Platform support

The table distinguishes native accelerated paths from compatible fallback
implementations. Quantized execution uses FP16 activations.

| Feature | CPU | Apple MPS | NVIDIA CUDA |
| --- | --- | --- | --- |
| Qwen3 dense checkpoints | Supported | Supported | Supported |
| Granite 3.1 sparse-MoE checkpoints | Supported | Supported | Supported |
| GPTQ INT4 (W4A16) | Not supported | Not supported | GPTQ-Marlin; Linux x86-64 and compute capability 7.5+ |
| GPTQ INT8 (W8A16) | Not supported | Not supported | GPTQ-Marlin; Linux x86-64 and compute capability 7.5+ |
| Regular PyTorch SDPA | Native | Native | Native |
| Variable-length FlashAttention | Not available | Not available | Native through the pinned vLLM wheel |
| Dense KV cache | Contiguous SDPA | Contiguous SDPA | Contiguous SDPA |
| Paged KV cache | Gather + SDPA | Gather + SDPA | Direct block-table FlashAttention |
| Granite MoE dispatch | Active-expert loop | Batched prefill and gathered decode | Batched prefill and gathered decode |
| CUDA graph decode | Not applicable | Not applicable | Attention components verified; end-to-end engine replay not yet integrated |
| Sampling, CLI, and HTTP API | Supported | Supported | Supported |

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/)
are required.

```bash
uv sync
```

This creates `.venv`, installs the project in editable mode with its development
dependencies, and reproduces the versions in `uv.lock`. Prefix project commands
with `uv run`; activating the environment is optional.

CUDA paged attention and GPTQ-Marlin execution use operators from the pinned
vLLM binary wheel. On Linux x86-64, install the operator dependency group with:

```bash
uv sync --group gptq
```

Keep that opt-in group selected when using CUDA paged attention or a GPTQ
checkpoint, for example `uv run --group gptq python -m mini_llm ...`.

For NVIDIA execution, install a CUDA-enabled PyTorch build and verify it can
see the GPU before loading a checkpoint:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Model directories

Place local Hugging Face model artifacts under `models/`:

```text
models/qwen3-0.6b/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Indexed checkpoints are also supported:

```text
models/qwen3-1.7b/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

The loader combines shard headers into one manifest and validates every tensor
name, shape, and dtype against the selected architecture before assigning any
weights. Granite uses the same required configuration, tokenizer, and
checkpoint artifacts.

## Run the Runtime

### One-shot generation

```bash
uv run python -m mini_llm \
  --model models/qwen3-0.6b \
  --prompt "Explain grouped-query attention." \
  --max-new-tokens 128 \
  --temperature 0
```

The prompt is raw user text. The runtime chooses the architecture-specific chat
template and tokenizer, streams generated text, and then prints latency and
cache metrics. Replace the model path with `models/granite-3.1-1b` to run
Granite.

### Interactive generation

```bash
uv run python -m mini_llm \
  --model models/qwen3-0.6b \
  --interactive \
  --max-new-tokens 128 \
  --temperature 0
```

Interactive mode loads the model once and accepts independent prompts until
`/quit`, `/exit`, or Control-D. Each prompt is independent; conversation
history is not carried between inputs.

### Important CLI options

| Option | Purpose |
| --- | --- |
| `--device auto\|cpu\|mps\|cuda` | Choose execution device |
| `--dtype auto\|float16\|bfloat16\|float32` | Choose model precision |
| `--max-seq-len 4096` | Set runtime context and cache limit |
| `--temperature`, `--top-k`, `--top-p`, `--seed` | Configure sampling |
| `--thinking` | Enable the Qwen3 thinking prompt; Granite rejects it |
| `--interactive` | Reuse one loaded model for multiple prompts |
| `--no-stream`, `--no-metrics` | Control terminal output |

`device=auto` selects CUDA when available, then MPS, and otherwise CPU.
`dtype=auto` selects FP16 on CUDA and MPS and FP32 on CPU. FP16 reduces
accelerator memory use, uses CUDA Tensor Cores, and provides more significand
bits than BF16. BF16 and FP32 remain available as explicit overrides, subject
to hardware support and VRAM capacity.

### Python API

```python
from mini_llm.engine import Engine
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage

engine = Engine.from_model_dir(
    "models/qwen3-0.6b",
    device="auto",
    dtype="auto",
    max_seq_len=4096,
    cache_backend="paged",  # or "dense"
)

for event in engine.generate(
    [ChatMessage("user", "Explain what a KV cache does.")],
    max_new_tokens=64,
    sampling=SamplingConfig(temperature=0),
):
    print(event.text_delta, end="", flush=True)
```

Each event contains the new stable text, complete emitted text, token index,
optional finish reason, and synchronized model-call duration. The first event
also contains the already-computed formatted prompt-token count.

Move a resident model without rereading its checkpoint:

```python
engine.to(device="cuda", dtype="float16")
```

The Python API also accepts indexed CUDA devices such as `cuda:0`. Moving the
model rebuilds device-specific runtime state and RoPE's derived FP32
frequencies. Dtype conversion changes the resident weights;
widening after a lossy downcast does not restore the original precision.

### HTTP server

```bash
uv run python -m mini_llm.serving.server \
  --model models/qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --device auto \
  --dtype auto \
  --max-seq-len 4096
```

The server loads one model before accepting requests and always uses one worker.
Model execution is currently serialized even though KV state is request-owned;
continuous batching is a later milestone. The served model name defaults to the
model-directory name and can be changed with `--served-model-name NAME`.

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "Answer briefly."},
      {"role": "user", "content": "What does a KV cache store?"}
    ],
    "temperature": 0,
    "max_completion_tokens": 64
  }'
```

OpenAI-compatible clients can use:

```text
base URL: http://127.0.0.1:8000/v1
model:    qwen3-0.6b
API key:  any placeholder if the client requires one
```

The local server does not inspect API keys and binds to `127.0.0.1` by default;
changing the host can expose the unauthenticated endpoint to the network.
FastAPI documentation is available at `http://127.0.0.1:8000/docs`. This
version supports text-only system, user, and assistant messages with one
non-streaming completion. HTTP streaming, tool calls, media, structured output,
penalties, stop strings, and authentication are deferred.

## How Inference Works

```text
raw messages
    → architecture-specific chat template
    → tokenizer
    → prompt token IDs
    → full-sequence prefill and K/V writes
    → sample first output token
    → cached single-token decode and one new K/V write
    → repeat until EOS, token limit, or context limit
```

Both architectures use token embeddings, pre-normalized decoder layers,
grouped-query causal attention, RoPE, residual connections, final RMSNorm, and
a vocabulary projection. Their main differences are:

| Detail | Qwen3-0.6B | Granite 3.1 1B-A400M |
| --- | --- | --- |
| Decoder layers | 28 | 24 |
| Feed-forward block | Dense SwiGLU | Top-8-of-32 sparse SwiGLU experts |
| Query/key normalization | Per-head Q/K RMSNorm | None |
| Attention scaling | `1 / sqrt(head_dim)` | Configured `1 / head_dim` |
| Embedding/output | Separate LM head | Tied embeddings, scaled logits |

### Granite MoE execution

All 32 experts remain resident, while each token executes only its selected
eight. Router projection and top-k selection run in FP32 because small reduced-
precision differences near the top-k boundary can select a different network.

- **CPU:** use the active-expert loop because it has lower dispatch overhead.
- **CUDA/MPS prefill:** group token activations by expert, pad groups to the
  busiest expert, and process `[32, max_assignments, 1024]` through batched
  matmuls.
- **CUDA/MPS decode:** gather the eight selected expert weight matrices and
  process the single token with two smaller batched matmuls.

Accelerator prefill keeps the larger gate/up projection in FP16 and widens the
smaller output projection and routing-weighted reduction to FP32. This preserves
most of the batching speedup while reducing backend-specific route divergence.

### Attention execution

Each attention layer first projects the residual stream into Q, K, and V,
normalizes Q/K where required by the architecture, and applies RoPE. New K/V
states are written into the selected request cache using device position IDs.
Grouped-query attention shares each K/V head across its corresponding query
heads, applies causal scaled dot-product attention, concatenates the attended
query heads, and projects the result back into the residual-stream width.

The runtime has two implementations of that same attention operation:

- **Regular PyTorch SDPA:** consumes contiguous
  `[batch, heads, tokens, head_dim]` tensors. The runtime repeats K/V heads for
  GQA and supplies an absolute-position causal mask during cached execution.
- **Variable-length FlashAttention on CUDA:** consumes flattened queries, physical K/V
  blocks, a block table, and the effective sequence length. It applies causal
  masking and GQA head mapping inside the kernel without materializing
  contiguous, head-repeated K/V tensors first.

```text
hidden states
    → Q/K/V projection
    → Q/K normalization and RoPE
    → K/V cache write
    → grouped-query causal attention
    → attention-output projection
```

### KV cache

The runtime owns request caches and passes them to models through one
backend-neutral interface. The default paged backend stores 16-token K/V
blocks in a global pool and gives each request one ordered block table shared
across decoder layers; a contiguous per-request dense backend remains
available for comparison.

- **Dense cache:** the cache exposes a contiguous K/V prefix and PyTorch SDPA
  performs causal attention. During CUDA graph capture, the view is rounded to
  a fixed 16-token bucket and the device-position mask hides unused entries, so
  one graph can replay at multiple lengths within that bucket.
- **Paged cache on CUDA:** vLLM `flash_attn_varlen_func` reads physical K/V
  blocks through the request's block table. It handles grouped-query attention
  directly and avoids gathering or repeating cached KV heads.
- **Paged cache on CPU/MPS:** the cache gathers its logical prefix into
  contiguous tensors and uses the same SDPA reference path as the dense
  backend.

See [Paged KV-cache](doc/paged-cache.md) for implementation details, the cache
layout, and CUDA benchmark results.

## Benchmarks

Run every applicable suite with one checkpoint load:

```bash
uv run python -m benchmarks --model models/granite-3.1-1b
```

Select suites and inputs explicitly:

```bash
uv run python -m benchmarks \
  --model models/granite-3.1-1b \
  --benchmark cache-decode moe-prefill end-to-end \
  --device cpu \
  --device cuda \
  --prompt-lengths 32 128 512 \
  --warmups 1 \
  --repeats 3 \
  --decode-tokens 16
```

The runner always uses FP16. It loads each checkpoint once, runs every selected
device, and moves the same engine without reloading. Available MPS and CUDA
devices are included by default; unavailable explicitly requested devices are
reported as errors. Load and transfer times are reported separately.

| Suite | Measurements |
| --- | --- |
| `cache-decode` | Cached versus uncached TPOT, throughput, cached speedup, cache memory, and logit agreement. Both paths run only for requested prompts up to 32 tokens. |
| `moe-prefill` | Full Granite prefill latency and throughput using the device's automatic expert method. |
| `end-to-end` | TTFT, prefill throughput, decode TPOT/throughput, output tokens, and cache memory through `Engine.generate`. |

Defaults are prompt lengths `32 128 512`, one untimed warmup, three measured
runs, and 16 decode tokens. Tables report medians. Warmups initialize lazy
kernels, allocator storage, and reusable caches. CUDA and MPS are synchronized
at timing boundaries because accelerator work is asynchronous.

TTFT includes prompt preparation, cache setup, prefill, and first-token
selection. Prefill is matrix-matrix-heavy and usually benefits strongly from
GPU execution. Batch-one decode is commonly memory-bandwidth-bound, so its
device speedup can be smaller.

## Development

Run the test suite:

```bash
uv run python -m pytest
```

Tests cover configuration and checkpoint validation, tokenizer/chat-template
agreement, explicit transformer equations, cached execution, sampling,
generation, CLI and HTTP behavior, benchmark orchestration, optional
Transformers references, and conditional real CUDA and MPS execution. Optional
tests skip cleanly when their model, dependency, or device is unavailable.

Focused learning examples:

```bash
uv run python -m examples.inspect_config models/qwen3-0.6b
uv run python -m examples.tokenizer_demo models/qwen3-0.6b "Hello"
uv run python -m examples.checkpoint_inspect models/qwen3-0.6b
uv run python -m examples.norm_demo
uv run python -m examples.rope_demo
uv run python -m examples.mlp_demo
uv run python -m examples.attention_demo
uv run python -m examples.moe_demo
uv run python -m examples.generation_demo models/qwen3-0.6b
```

## Current Limitations

- One active request and batch size one.
- Synchronous, non-streaming HTTP responses.
- CUDA paged attention requires the optional pinned vLLM operator group.
- No prefix sharing, sliding-window eviction, or CPU cache offload.
- No concurrent batching.
- Only Qwen3 and Granite 3.1 MoE architectures.
