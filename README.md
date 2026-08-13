# miniVLLM

miniVLLM is a compact inference system that follows the core layered architecture of
vLLM v1. It provides a stable system foundation for custom scheduling, memory
management, and kernel optimizations. The existing CUDA kernels and NCU profiling
entry points are preserved and integrated into a runnable multi-process request path.

The current focus is the core system architecture rather than full vLLM compatibility:

- API, EngineCore, and Worker have explicit process boundaries.
- `LLMEngine` provides a synchronous streaming interface, while `AsyncLLMEngine`
  provides an asynchronous streaming interface.
- Each data-parallel rank owns an independent EngineCore that performs token-budget
  scheduling, chunked prefill, and continuous batching.
- When DP > 1, a dedicated Coordinator balances requests across ranks. Each rank's
  Worker owns an independent paged KV cache.
- Scheduler output is flattened into one `GPUInputBatch`, and the Worker invokes the
  ModelRunner only once per scheduling step.
- Mixed prefill/decode batches, local Llama safetensors, the 123M Interpretable GPT,
  and Hugging Face tokenizers are supported.
- Real Llama Q/K/V tensors enter custom Flash Prefill and Paged Decode kernels through
  a PyTorch CUDA tensor ABI.
- Decode-only Llama forward passes support CUDA Graph replay with capture-size buckets.
- `/metrics` reports batch, token, prefill, decode, and Worker execution statistics.
  Generation results include TTFT, ITL, and end-to-end latency.
- Optional trace events expose scheduling decisions, physical KV blocks, execution
  mode, and sampled tokens for reproducible systems visualizations.
- The existing `kernels/`, `bench/`, and NCU scripts remain the entry points for kernel
  performance analysis.

By default, the `reference` Worker uses a deterministic bootstrap ModelRunner so the
system can be exercised without model files. Its output does not represent a real LLM.
When given a supported local model directory, the Worker reads `model_type`, loads the
actual model layers and weights, and allocates a rank-local paged KV cache.

## Process Architecture

```text
API process
  └─ AsyncLLMEngine / LLMEngine
       └─ DP Coordinator process (only when DP > 1)
            ├─ EngineCore process (DP rank 0)
            │    └─ Worker process (GPU 0 + rank-local KV cache)
            └─ EngineCore process (DP rank 1)
                 └─ Worker process (GPU 1 + rank-local KV cache)
```

Protocol conversion happens only at the API layer. The Coordinator only selects a
rank, and the corresponding EngineCore is the sole owner of that request's scheduling
state. A Worker only executes `WorkerBatch` objects and never changes request state on
its own. This preserves the vLLM v1 separation between the DP Coordinator, EngineCore,
and Executor.

See [docs/architecture.md](docs/architecture.md) for object relationships, request
state transitions, and the recommended source-reading order.

## Quick Start

The default bootstrap path has no third-party Python dependencies.

```bash
python -m examples.basic
python -m unittest discover -v
```

Start the minimal OpenAI-compatible completions API:

```bash
python -m minivllm.entrypoints.api_server \
  --port 8000 \
  --data-parallel-size 2 \
  --backend reference
```

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","max_tokens":8,"stream":false}'
```

Runtime metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

## 123M Interpretable GPT

The `interpretable_gpt` runner connects the ground-up 12-layer GPT checkpoint to the
same Scheduler, mixed-batch attention ABI, paged KV cache, and CUDA Graph dispatcher
used by the Llama runner. It preserves the checkpoint's manual first LayerNorm,
separate biased Q/K/V projections, 12-head RoPE, GELU MLP, and tied token embeddings.

The GPT repository includes `export_minivllm.py`, which converts the original `.pt`
state dict into a self-contained safetensors model directory with GPT-2 tokenizer
files. After export:

```bash
python -m examples.interpretable_gpt \
  --model /models/interpretable-gpt-123m \
  --prompt "The future of AI systems is" \
  --device cuda:0 \
  --dtype float16 \
  --attention-backend custom
```

Validate legacy full-sequence logits against paged incremental decoding:

```bash
python scripts/validate_interpretable_gpt.py \
  --model-dir /path/to/GPT/src
```

Capture a deterministic, website-ready execution trace:

```bash
python scripts/capture_generation_trace.py \
  --model /models/interpretable-gpt-123m \
  --output artifacts/portfolio/interpretable_gpt.json
```

## Local Llama Models

The model path uses optional dependencies and reads files only from a local directory:

```bash
pip install -e ".[model]"
python -m minivllm.entrypoints.api_server \
  --model /models/llama \
  --device cuda:0 \
  --dtype float16 \
  --attention-backend custom \
  --cudagraph-mode full_decode_only
```

The minimal Llama runner currently supports standard RoPE, RMSNorm, GQA/MHA, SwiGLU,
greedy sampling, and temperature sampling. Each layer writes K/V tensors to
`physical_block * block_size + offset`, and mixed batches read context through each
request's block table. On CUDA devices, `auto` is equivalent to `custom`; on CPU, it
uses the PyTorch reference implementation with the same paged-attention semantics.
Select `torch` explicitly for CUDA comparison experiments.

On first use, the custom backend compiles `runtime/torch_attention.cpp` and
`runtime/torch_attention_cuda.cu` as a PyTorch extension. This requires the CUDA
Toolkit, a compiler compatible with the installed PyTorch version, and Ninja. PyTorch
caches the compiled artifacts for subsequent Worker processes.

## CUDA Kernel Runtime

The existing kernel benchmarks remain unchanged:

```bash
make
./scripts/profile_paged_kv_cache.sh 512 8
```

The bootstrap Worker also retains a ctypes C ABI. It persistently owns a synthetic KV
cache and accepts the slot mapping, block table, and flattened batch metadata produced
by EngineCore:

```bash
make native
python -m minivllm.entrypoints.api_server --backend cuda
```

`MINIVLLM_OPS_LIBRARY` can point to the generated `.so`, `.dll`, or `.dylib`. The
current C ABI is a correctness-first reference launch. The primary optimization and
NCU targets remain `kernels/flash_attention/flash_attention_prefilling.cu` and
`kernels/paged_kv_cache/paged_kv_cache.cu`. Replacing the kernels behind the C ABI does
not affect the system layer.

The real Llama path does not use ctypes or copy Q/K/V to the host. The Torch extension
receives these CUDA tensors directly:

```text
Q                    [scheduled_tokens, query_heads, head_dim]
K/V                  [scheduled_tokens, kv_heads, head_dim]
layer KV cache       [physical_slots, kv_heads, head_dim]
slot_mapping         [scheduled_tokens]
block_tables         [requests, max_blocks]
positions/seq_lens   token/request metadata
```

At each layer, `append_kv_` first writes the real K/V values. Then
`flash_prefill_out` processes all prefill tokens, and `paged_decode_out` processes the
decode requests. Both kernels run on the same CUDA stream and write to the same
attention output tensor.

## CUDA Graph Status

The system provides `BatchDescriptor`, decode capture-size buckets, a unified
dispatcher, and eager fallback:

```bash
python -m minivllm.entrypoints.api_server \
  --model /models/llama \
  --device cuda:0 \
  --attention-backend custom \
  --cudagraph-mode full_decode_only
```

For each capture size, the real Llama path maintains static input, position, slot
mapping, block table, and active-mask tensors. On the first matching batch, it performs
warm-up and captures the embedding, all decoder layers, custom Paged Attention, and LM
head. Subsequent batches update only the static tensors before replay. Padding rows use
KV scratch slots and cannot overwrite active requests. Mixed or prefill batches, CPU
execution, and the `torch` attention backend fall back to eager execution. The actual
execution mode is recorded in the batch metrics.

## Project Layout

```text
minivllm/
  engine/             # facade, DP Coordinator, EngineCore, IPC client
  worker/             # Worker, GPUInputBatch, ModelRunner, CUDA Graph dispatcher
  model_executor/     # Llama/GPT, attention backend, paged KV tensors, graph state
  entrypoints/        # API server process entry point
  cache.py            # physical block pool and per-request block tables
  scheduler.py        # v1-style token scheduler
runtime/
  minivllm_ops.cu             # bootstrap ctypes CUDA C ABI
  torch_attention.cpp         # real Q/K/V Torch extension entry point
  torch_attention_cuda.cu     # Flash Prefill / Paged Decode kernels
kernels/              # existing CUDA kernels and benchmarks
tests/                 # memory, scheduling, multiprocessing, and async concurrency tests
```

## Intentionally Omitted

- Tensor parallelism and pipeline parallelism
- Prefix caching, LoRA, and speculative decoding
- Llama RoPE scaling, quantized weights, and additional model architectures
- Deep attention-kernel optimization for Tensor Cores, long contexts, and different
  `head_dim` values
- Distributed-node coordination and production-grade fault recovery
- Full OpenAI API and vLLM parameter compatibility

These components are not represented by empty placeholder classes. The architecture
document describes the available extension points and the recommended implementation
order.
