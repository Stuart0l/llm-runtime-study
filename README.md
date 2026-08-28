# Mini Qwen3 Inference Runtime

This project is a small, study-oriented text-generation runtime for the
Qwen3-0.6B Safetensors checkpoint. The transformer execution path is
implemented directly with PyTorch; it does not use Transformers to load or run
the model. The trained tokenizer algorithm is reused through `tokenizers`.

The runtime supports one active request with batch size one. It performs one
prompt prefill followed by single-token decoding through a preallocated KV
cache, and can run on Apple MPS or CPU.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Place a Qwen3-0.6B model directory at `models/qwen3-0.6b`. This version expects
at least:

```text
models/qwen3-0.6b/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Only a single-file, unsharded Safetensors checkpoint is currently supported.

## Generate text

```bash
python -m mini_llm \
  --model models/qwen3-0.6b \
  --prompt "Explain grouped-query attention." \
  --max-new-tokens 128 \
  --temperature 0
```

The prompt is raw user text. The runtime applies Qwen3's chat template and
tokenizes it internally. Generated text streams to the terminal, followed by a
benchmark summary.

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

`device=auto` selects MPS when available and otherwise CPU. `dtype=auto`
selects FP16 on MPS and FP32 on CPU. MPS FP16 is intentional: for Qwen's
normalized inference activations its finer significand precision is generally
more useful than BF16's larger exponent range. BF16 remains available as an
explicit override.

## Python API

```python
from mini_llm.engine import Engine
from mini_llm.sampling import SamplingConfig

engine = Engine.from_model_dir(
    "models/qwen3-0.6b",
    device="auto",
    dtype="auto",
    max_seq_len=4096,
)

for event in engine.generate(
    "Explain what a KV cache does.",
    max_new_tokens=64,
    sampling=SamplingConfig(temperature=0),
):
    print(event.text_delta, end="", flush=True)
```

Each event contains the new stable text in `text_delta`, the complete emitted
text in `text`, the generated token index, an optional finish reason, and the
synchronized model-call duration when generation is run through `Engine`. The
first event also carries the formatted prompt-token count already calculated by
the generation loop, so benchmark consumers do not tokenize the prompt twice.

## Execution architecture

```text
raw user prompt
      │
      ▼
Qwen chat formatting → tokenizer → prompt token IDs
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

The model itself is composed as:

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

## KV cache

Each layer owns preallocated key and value tensors:

```text
[1, num_kv_heads, capacity, head_dim]
```

For Qwen3-0.6B this is `[1, 8, capacity, 128]`. The cache stores the original
eight grouped-query K/V heads; expansion to sixteen query heads happens only
inside attention computation.

Cache memory is:

```text
layers × 2(K,V) × KV heads × capacity × head dimension × bytes per value
```

Resetting changes only the logical length, so the allocated tensors can be
reused by the next request.

## Benchmark metrics

The CLI reports:

- Model and tokenizer load time.
- Formatted prompt-token count.
- Generated-token count.
- Allocated cache size and capacity.
- End-to-end time to first token (TTFT).
- Decode tokens per second, excluding the token produced by prefill.
- The termination reason.

MPS operations are asynchronous. Timing therefore synchronizes the device at
the measurement boundaries; without synchronization, the CPU clock would stop
before GPU work completed. TTFT includes prompt preparation, cache setup,
prefill, first-token selection, and incremental decoding, making it more useful
to users than a nearly identical standalone prefill-latency number.

Prefill and decode measure different workloads. Prefill applies model weights
to many prompt tokens in parallel and benefits strongly from GPU matrix-matrix
operations. Batch-one decode applies the weights to one token at a time and is
usually memory-bandwidth-bound, so its CPU-to-MPS speedup can be much smaller.

## Tests and demonstrations

Run the complete suite:

```bash
python -m unittest discover -s tests -v
```

The suite includes a conditional real MPS test and an optional Hugging Face
correctness oracle. They skip cleanly when MPS, Transformers, or the local
checkpoint is unavailable.

Focused educational demonstrations live in `examples/`:

```bash
python -m examples.inspect_config models/qwen3-0.6b
python -m examples.tokenizer_demo models/qwen3-0.6b
python -m examples.checkpoint_inspect models/qwen3-0.6b
python -m examples.cache_demo models/qwen3-0.6b
python -m examples.generation_demo models/qwen3-0.6b
python -m examples.engine_demo models/qwen3-0.6b
```

## Current scope

Implemented:

- Qwen3 architecture only.
- Single request and batch size one.
- Single-file Safetensors loading without Transformers.
- RMSNorm, Q/K normalization, RoPE, SwiGLU, and grouped-query attention.
- Dense preallocated KV cache.
- Greedy, temperature, top-k, and top-p sampling.
- Seeded generation, streaming text, CPU execution, and Apple MPS execution.

Deferred:

- HTTP serving and concurrent batching.
- Paged attention.
- Quantization.
- Sharded checkpoints.
- Sliding-window cache eviction.
- Other model architectures.
