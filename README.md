# Mini LLM Inference Runtime

This project is a small, study-oriented text-generation runtime for Qwen3
dense checkpoints and IBM Granite 3.1 sparse-MoE checkpoints. The transformer
execution path is implemented directly with PyTorch; it does not use
Transformers to load or run a model. The trained tokenizer algorithms are
reused through `tokenizers`.

The runtime supports one active request with batch size one. It performs one
prompt prefill followed by single-token decoding through a preallocated KV
cache, can run on Apple MPS or CPU, and can expose synchronous OpenAI-compatible
Chat Completions over HTTP.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Place a supported Hugging Face model directory under `models/`. A single-file
Qwen checkpoint contains at least:

```text
models/qwen3-0.6b/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The Granite model uses the same four required files in a directory such as
`models/granite-3.1-1b`. The runtime also supports standard indexed
Safetensors checkpoints such as Qwen3-1.7B:

```text
models/qwen3-1.7b/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

The runtime reads every shard header into one combined manifest. The registered
architecture schema then checks all tensor names, shapes, and dtypes before
model weights are loaded. Shard indexes are treated as trusted model files.

## Generate text

```bash
python -m mini_llm \
  --model models/qwen3-0.6b \
  --prompt "Explain grouped-query attention." \
  --max-new-tokens 128 \
  --temperature 0
```

The prompt is raw user text. The runtime selects the model's chat template and
tokenizer automatically. Generated text streams to the terminal, followed by a
benchmark summary. Substitute `models/granite-3.1-1b` in the same command to
run Granite.

To load the model once and enter multiple independent prompts, use interactive
mode:

```bash
python -m mini_llm \
  --model models/qwen3-0.6b \
  --interactive \
  --max-new-tokens 128 \
  --temperature 0
```

The model weights and tokenizer remain loaded between prompts. Each prompt
starts a fresh generation request and KV cache; the runtime does not yet carry
conversation history from one prompt to the next. Enter `/quit` or `/exit`, or
send EOF with Control-D, to stop. `--prompt` may be supplied with
`--interactive` to provide the first prompt before the input loop begins.

Useful options:

```text
--device auto|cpu|mps
--dtype auto|float16|bfloat16|float32
--max-seq-len 4096
--temperature 0.8
--top-k 20
--top-p 0.9
--seed 42
--thinking
--interactive
--no-stream
--no-metrics
```

`--thinking` is a Qwen3 chat-template option; Granite rejects it explicitly.

`device=auto` selects MPS when available and otherwise CPU. `dtype=auto`
selects FP16 on MPS and FP32 on CPU. MPS FP16 is intentional: its finer
significand precision avoids some accumulated BF16 rounding observed in these
models. BF16 remains available as an explicit override.

Granite computes the router projection and top-k decision in FP32. Attention
and decode experts use the selected model dtype. Padded MPS prefill keeps its
larger gate/up batched projection in that dtype and widens only the smaller
output projection to FP32, preventing backend-specific rounding from changing
later expert routes without giving up most of the batched-prefill speedup.

## Python API

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

Move the already-loaded model without reading its checkpoint again:

```python
engine.to(device="mps", dtype="float16")
```

Moving invalidates the device-specific KV cache and rebuilds RoPE's derived
FP32 frequencies on the destination. Dtype conversion changes the resident
weights: widening after a lossy downcast does not restore the checkpoint's
original precision.

Each event contains the new stable text in `text_delta`, the complete emitted
text in `text`, the generated token index, an optional finish reason, and the
synchronized model-call duration when generation is run through `Engine`. The
first event also carries the formatted prompt-token count already calculated by
the generation loop, so benchmark consumers do not tokenize the prompt twice.

## HTTP server

Start one local server process with:

```bash
python -m mini_llm.server \
  --model models/qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000 \
  --device auto \
  --dtype auto \
  --max-seq-len 4096
```

The model and tokenizer load once before the server accepts requests. The
served model name defaults to the model directory name (`qwen3-0.6b` here).
Use `--served-model-name NAME` to override it. The command always starts one
worker because all requests share the model's mutable KV cache.

Send a synchronous Chat Completions request:

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

For an existing OpenAI-compatible client or UI, configure:

```text
base URL: http://127.0.0.1:8000/v1
model:    qwen3-0.6b
API key:  any placeholder if the client requires one
```

The server itself does not inspect an API key. It binds only to localhost by
default; choosing another host can expose the unauthenticated server to the
network. FastAPI's interactive schema is available at
`http://127.0.0.1:8000/docs` while the server is running.

This version supports text-only system, user, and assistant history, one
completion, and ordinary JSON responses. HTTP streaming, tools, media,
developer messages, penalties, stop strings, structured output, and
authentication are not implemented. Concurrent valid requests wait in order
and run one at a time so they never overwrite each other's KV cache.

## Execution architecture

```text
raw user prompt
      │
      ▼
architecture chat formatting → tokenizer → prompt token IDs
      │
      ▼
one full-sequence prefill
      │
      ├── writes K/V for every prompt position into each layer cache
      │
      ▼
last-position vocabulary logits → sampling → first output token
      │
      ▼
single-token decode → append one K/V position → next token
      │
      └───────────────────────────────────────────────┐
                                                      │
                repeat until EOS, token limit, or context limit
```

Qwen uses dense SwiGLU decoder layers:

```text
token IDs
  → token embeddings
  → 28 decoder layers
      → RMSNorm
      → grouped-query causal attention + RoPE + Q/K normalization
      → residual connection
      → RMSNorm
      → SwiGLU feed-forward network
      → residual connection
  → final RMSNorm
  → language-model head
  → vocabulary logits
```

Granite uses 24 decoder layers without Q/K normalization. Each layer replaces
the dense MLP with a top-8-of-32 packed SwiGLU expert block, scales embeddings
by 12, scales attention and MoE residual branches by 0.22, ties the vocabulary
projection to the embedding matrix, and divides final logits by 6.

All 32 experts stay loaded, but each token selects only eight. MPS uses two
batching strategies:

- **Prefill:** group token activations by expert and pad each group to the size
  of the busiest one, producing `[32, max_assignments, 1024]`. Two batched
  matrix multiplications run the experts; saved positions map their weighted
  outputs back to the original tokens.
- **Decode:** one token activates only eight experts, so gather those eight
  weight matrices and run them as two smaller batched matrix multiplications.

CPU keeps the expert loop because it is faster there.

## KV cache

Each layer owns preallocated key and value tensors:

```text
[1, num_kv_heads, capacity, head_dim]
```

For Qwen3-0.6B this is `[1, 8, capacity, 128]`; for Granite it is
`[1, 8, capacity, 64]`. The cache stores the original grouped-query K/V heads;
expansion to query-head count happens only inside attention computation.

Cache memory is:

```text
layers × 2(K,V) × KV heads × capacity × head dimension × bytes per value
```

Resetting changes only the logical length, so the allocated tensors can be
reused by the next request.

## Benchmarks

Run every applicable suite with one checkpoint load:

```bash
python -m benchmarks --model models/granite-3.1-1b
```

Or select suites and inputs explicitly:

```bash
python -m benchmarks \
  --model models/granite-3.1-1b \
  --benchmark cache-decode moe-prefill end-to-end \
  --device cpu \
  --device mps \
  --prompt-lengths 32 128 512 \
  --warmups 1 \
  --repeats 3 \
  --decode-tokens 16
```

The runner always uses FP16. It loads each model only once, completes every CPU
suite, and then moves the same resident engine to MPS without rereading
Safetensors. Model load and device-transfer time are printed separately. MPS
is included by default when available; an explicitly requested unavailable MPS
device is an error.

The suites measure:

- `cache-decode`: cached single-token decode versus recomputing the complete
  prefix, including TPOT, throughput, speedup, cache memory, and final-logit
  agreement. Because this suite compares the two algorithms, both paths run
  only for requested prompt lengths up to 32 tokens; longer cases are omitted.
- `moe-prefill`: full Granite prefill over several prompt lengths. CPU uses the
  active-expert loop and MPS uses padded expert batching automatically.
- `end-to-end`: TTFT, synchronized prefill throughput, decode TPOT and
  throughput, generated tokens, and cache memory through `Engine.generate`.

One untimed warmup and three measured repetitions are the defaults. Tables show
median times. Input construction, checkpoint loading, cache allocation, and
correctness comparisons remain outside measured regions. MPS timing explicitly
synchronizes at measurement boundaries because accelerator work is otherwise
asynchronous.

TTFT includes prompt preparation, cache setup, prefill, and first-token
selection. Prefill is matrix-matrix-heavy and benefits strongly from MPS;
batch-one decode is usually memory-bandwidth-bound, so its CPU-to-MPS speedup
can be much smaller.

## Tests and demonstrations

Run the complete suite:

```bash
python -m unittest discover -s tests -v
```

The suite includes a conditional real MPS test and an optional Hugging Face
correctness oracle for both architectures. It also sends conditional CLI and
HTTP requests through the real local checkpoints. These tests skip cleanly
when MPS, Transformers, or a local checkpoint is unavailable.

Granite coverage includes its 218-tensor checkpoint, official prompt tokens,
explicit router and packed-expert equations, cached versus uncached logits,
greedy output against Transformers, and CPU/MPS FP16 greedy agreement. The MPS
regression exercises both padded prefill and batched single-token expert paths.

Focused educational demonstrations live in `examples/`:

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

## Current scope

Implemented:

- Qwen3 dense and Granite 3.1 sparse-MoE architectures.
- Single request and batch size one.
- Single-file and indexed sharded Safetensors loading without Transformers.
- RMSNorm, optional Q/K normalization, RoPE, SwiGLU, grouped-query attention,
  and top-k packed-expert routing.
- Dense preallocated KV cache.
- Greedy, temperature, top-k, and top-p sampling.
- Seeded generation, streaming text, CPU execution, and Apple MPS execution.
- Synchronous OpenAI-compatible Chat Completions over FastAPI.

Deferred:

- HTTP streaming and concurrent batching.
- Paged attention.
- Quantization.
- Sliding-window cache eviction.
- Additional model architectures.
