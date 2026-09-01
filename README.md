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
| Cache | Preallocated dense KV cache with prefill and single-token decode |
| Serving | Synchronous OpenAI-compatible Chat Completions through FastAPI |

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For NVIDIA execution, install a CUDA-enabled PyTorch build and verify it can
see the GPU before loading a checkpoint:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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
python -m mini_llm \
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
python -m mini_llm \
  --model models/qwen3-0.6b \
  --interactive \
  --max-new-tokens 128 \
  --temperature 0
```

Interactive mode loads the model once and accepts independent prompts until
`/quit`, `/exit`, or Control-D. Each prompt starts a fresh chat and resets the
logical KV-cache length; conversation history is not carried between inputs.

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
model invalidates its device-specific KV cache and rebuilds RoPE's
derived FP32 frequencies. Dtype conversion changes the resident weights;
widening after a lossy downcast does not restore the original precision.

### HTTP server

```bash
python -m mini_llm.server \
  --model models/qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --device auto \
  --dtype auto \
  --max-seq-len 4096
```

The server loads one model before accepting requests and always uses one worker
because requests share a mutable KV cache. Concurrent valid requests wait and
execute one at a time. The served model name defaults to the model-directory
name and can be changed with `--served-model-name NAME`.

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

### KV cache

Each layer owns preallocated key and value tensors:

```text
[1, num_kv_heads, capacity, head_dim]
```

Qwen3-0.6B uses `[1, 8, capacity, 128]`; Granite uses
`[1, 8, capacity, 64]`. The cache stores the original GQA key/value heads and
expands them to query-head count only during attention.

```text
cache bytes = layers × 2(K,V) × KV heads × capacity × head_dim × bytes/value
```

Resetting changes only logical length, allowing the same allocation to serve a
later request. Moving the engine to another device or dtype releases the old
cache because its storage is no longer compatible.

## Benchmarks

Run every applicable suite with one checkpoint load:

```bash
python -m benchmarks --model models/granite-3.1-1b
```

Select suites and inputs explicitly:

```bash
python -m benchmarks \
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
python -m unittest discover -s tests -v
```

Tests cover configuration and checkpoint validation, tokenizer/chat-template
agreement, explicit transformer equations, cached execution, sampling,
generation, CLI and HTTP behavior, benchmark orchestration, optional
Transformers references, and conditional real CUDA and MPS execution. Optional
tests skip cleanly when their model, dependency, or device is unavailable.

Focused learning examples:

```bash
python -m examples.inspect_config models/qwen3-0.6b
python -m examples.tokenizer_demo models/qwen3-0.6b "Hello"
python -m examples.checkpoint_inspect models/qwen3-0.6b
python -m examples.norm_demo
python -m examples.rope_demo
python -m examples.mlp_demo
python -m examples.attention_demo
python -m examples.moe_demo
python -m examples.generation_demo models/qwen3-0.6b
```

## Current Limitations

- One active request and batch size one.
- Synchronous, non-streaming HTTP responses.
- Dense KV cache rather than paged attention.
- No quantization or sliding-window eviction.
- No concurrent batching.
- Only Qwen3 and Granite 3.1 MoE architectures.
