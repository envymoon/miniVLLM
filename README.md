# miniVLLM

这是一个遵循 vLLM v1 核心分层方式的精简推理系统，目标是在稳定系统骨架上持续加入
自定义调度、内存管理和算子优化。项目保留原有 CUDA 算子和 NCU profiling 入口，
并在它们之上提供可运行的多进程请求链路。

当前重点是系统骨架，而不是兼容完整 vLLM：

- API、EngineCore、Worker 是清晰的进程边界；
- `LLMEngine` 提供同步流式接口，`AsyncLLMEngine` 提供异步流式接口；
- 每个 DP rank 有独立 EngineCore，执行 token-budget 调度、chunked prefill 和
  continuous batching；
- DP>1 时由独立 Coordinator 负载均衡；每个 rank 的 Worker 拥有独立分页 KV Cache；
- Scheduler 输出被压平为一个 `GPUInputBatch`，Worker 每轮只调用一次 ModelRunner；
- 支持 mixed prefill/decode batch、本地 Llama safetensors 和 Hugging Face tokenizer；
- 真实 Llama Q/K/V 通过 PyTorch CUDA tensor ABI 进入自定义 Flash Prefill 与
  Paged Decode kernel；
- 纯 decode 的真实 Llama forward 支持按 capture bucket 进行 CUDA Graph replay；
- `/metrics` 提供 batch/token/prefill/decode/Worker 执行时间统计，生成结果携带
  TTFT、ITL 和端到端延迟；
- 原有 `kernels/`、`bench/` 和 NCU 脚本保持为算子性能分析入口。

默认 `reference` Worker 使用确定性的 bootstrap ModelRunner，便于在不准备模型文件时
执行系统回归，输出不代表真实 LLM。传入本地 Llama 模型目录后，Worker 会加载真实模型
层、safetensors 权重和 rank-local 分页 KV Cache。

## 进程架构

```text
API process
  └─ AsyncLLMEngine / LLMEngine
       └─ DP Coordinator process (only when DP > 1)
            ├─ EngineCore process (DP rank 0)
            │    └─ Worker process (GPU 0 + rank-local KV cache)
            └─ EngineCore process (DP rank 1)
                 └─ Worker process (GPU 1 + rank-local KV cache)
```

请求只在 API 层做协议转换。Coordinator 只选择 rank；对应 EngineCore 是该请求调度
状态的唯一所有者。Worker 只执行 `WorkerBatch`，不会自行改变请求状态。这与 vLLM v1
的 DP Coordinator / EngineCore / Executor 分离保持一致。

更详细的对象关系、请求状态转换和源码阅读顺序见
[docs/architecture.md](docs/architecture.md)。

## 快速运行

默认 bootstrap 路径没有第三方 Python 依赖。

```bash
python -m examples.basic
python -m unittest discover -v
```

启动 OpenAI 风格的最小 completions API：

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

运行时指标：

```bash
curl http://127.0.0.1:8000/metrics
```

## 本地 Llama 模型

模型路径使用可选依赖，并且只从本地目录读取文件：

```bash
pip install -e ".[model]"
python -m minivllm.entrypoints.api_server \
  --model /models/llama \
  --device cuda:0 \
  --dtype float16 \
  --attention-backend custom \
  --cudagraph-mode full_decode_only
```

最小 Llama runner 当前支持标准 RoPE、RMSNorm、GQA/MHA、SwiGLU、greedy 和
temperature sampling。每层 K/V tensor 按 `physical_block * block_size + offset` 写入，
mixed batch 根据请求的 block table 读取上下文。`auto` 在 CUDA 设备上等价于 `custom`，
在 CPU 上使用相同分页语义的 PyTorch reference；`torch` 可显式用于 CUDA 对照实验。

自定义 backend 首次启动时通过 PyTorch extension 编译 `runtime/torch_attention.cpp` 和
`runtime/torch_attention_cuda.cu`，需要 CUDA Toolkit、与 PyTorch 匹配的编译器及 Ninja。
编译产物由 PyTorch 缓存，后续 Worker 直接加载。

## CUDA 算子运行时

原有算子 benchmark 不变：

```bash
make
./scripts/profile_paged_kv_cache.sh 512 8
```

bootstrap Worker 另外保留 ctypes C ABI。它持久持有合成 KV Cache，并接受 EngineCore
产生的 slot mapping、block table 和压平后的 batch metadata：

```bash
make native
python -m minivllm.entrypoints.api_server --backend cuda
```

也可用 `MINIVLLM_OPS_LIBRARY` 指向生成的 `.so`、`.dll` 或 `.dylib`。当前 C ABI 是
correctness-first 的参考 launch；`kernels/flash_attention/flash_attention_prefilling.cu` 和
`kernels/paged_kv_cache/paged_kv_cache.cu` 仍是优化与 NCU 的主战场。把优化 kernel
替换进 C ABI 不会影响系统层。

真实 Llama 不经过 ctypes，也不复制 Q/K/V 到 host。Torch extension 直接接收以下 CUDA
tensor：

```text
Q                    [scheduled_tokens, query_heads, head_dim]
K/V                  [scheduled_tokens, kv_heads, head_dim]
layer KV cache       [physical_slots, kv_heads, head_dim]
slot_mapping         [scheduled_tokens]
block_tables         [requests, max_blocks]
positions/seq_lens   token/request metadata
```

每层先用 `append_kv_` 写入真实 K/V，然后 `flash_prefill_out` 处理所有 prefill token，
`paged_decode_out` 处理 decode 请求。两个 kernel 使用同一 CUDA stream，并写入同一个
attention output tensor。

## CUDA Graph 状态

系统已经提供 `BatchDescriptor`、decode capture-size bucket、统一 dispatcher 和 eager
fallback：

```bash
python -m minivllm.entrypoints.api_server \
  --model /models/llama \
  --device cuda:0 \
  --attention-backend custom \
  --cudagraph-mode full_decode_only
```

真实 Llama 为每个 capture size 维护静态 input、position、slot mapping、block table 和
active mask。首次命中执行 warm-up 并捕获 embedding、全部 decoder layers、自定义 Paged
Attention 与 LM head；后续只更新静态 tensor 并 replay。padding 行使用 KV scratch slots，
不会覆盖活动请求。mixed/prefill batch、CPU 和 `torch` attention backend 会回退 eager，
实际模式记录在 batch metrics 中。

## 目录

```text
minivllm/
  engine/             # facade、DP Coordinator、EngineCore、IPC client
  worker/             # Worker、GPUInputBatch、ModelRunner、CUDA Graph dispatcher
  model_executor/     # Llama、attention backend、分页 KV tensor 与 graph state
  entrypoints/        # API server 进程入口
  cache.py            # 物理块池和请求 block table
  scheduler.py        # v1 风格 token scheduler
runtime/
  minivllm_ops.cu             # bootstrap ctypes CUDA C ABI
  torch_attention.cpp         # 真实 Q/K/V Torch extension 入口
  torch_attention_cuda.cu     # Flash Prefill / Paged Decode kernels
kernels/              # 原有 CUDA 算子与 benchmark
tests/                 # 内存、调度、多进程与异步并发测试
```

## 当前有意省略的部分

- tensor parallel / pipeline parallel；
- prefix caching、LoRA、speculative decoding；
- Llama RoPE scaling、量化权重和非 Llama 模型；
- 针对 Tensor Core、长上下文和不同 head_dim 的 attention kernel 深度优化；
- 分布式节点协调和生产级故障恢复；
- 完整 OpenAI API 与 vLLM 参数兼容层。

这些部分没有用空壳类伪装实现。系统中的扩展缝隙和推荐实现顺序记录在架构文档中。
