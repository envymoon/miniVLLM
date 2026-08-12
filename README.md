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
- CUDA C ABI 以一次 launch 处理 batch 中所有 attention 请求；
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
  --dtype float16
```

最小 Llama runner 当前支持标准 RoPE、RMSNorm、GQA/MHA、SwiGLU、greedy 和
temperature sampling。每层 K/V tensor 按 `physical_block * block_size + offset` 写入，
mixed batch 根据请求的 block table 读取上下文。CUDA 上的 PyTorch SDPA 可选择 Flash
Attention 后端；把仓库内定制 Flash/Paged kernel 直接连接到真实 Q/K/V ABI 是下一阶段。

## CUDA 算子运行时

原有算子 benchmark 不变：

```bash
make
./scripts/profile_paged_kv_cache.sh 512 8
```

另外提供了 Worker 可加载的稳定 C ABI。它持久持有 KV Cache，并接受 EngineCore 产生
的 slot mapping、block table 和压平后的 batch metadata：

```bash
make native
python -m minivllm.entrypoints.api_server --backend cuda
```

也可用 `MINIVLLM_OPS_LIBRARY` 指向生成的 `.so`、`.dll` 或 `.dylib`。当前 C ABI 是
correctness-first 的参考 launch；`kernels/flash_attention/flash_attention_prefilling.cu` 和
`kernels/paged_kv_cache/paged_kv_cache.cu` 仍是优化与 NCU 的主战场。把优化 kernel
替换进 C ABI 不会影响系统层。

## CUDA Graph 状态

系统已经提供 `BatchDescriptor`、decode capture-size bucket、统一 dispatcher 和 eager
fallback：

```bash
python -m minivllm.entrypoints.api_server \
  --backend cuda \
  --cudagraph-mode full_decode_only
```

CUDA bootstrap runtime 为纯 decode bucket 维护静态 device buffer，并缓存对应的
`cudaGraphExec_t`；输入数据更新后直接 replay。mixed/prefill batch 会回退 eager。真实
Llama attention 仍包含动态上下文 gather，所以 Llama ModelRunner 暂时也明确回退 eager，
并在 batch metrics 中报告实际执行模式。后续把 graph-safe 真实 Q/K/V kernel 接到同一
Manager 即可，不需要修改 Scheduler。

## 目录

```text
minivllm/
  engine/             # facade、DP Coordinator、EngineCore、IPC client
  worker/             # Worker、GPUInputBatch、ModelRunner、CUDA Graph dispatcher
  model_executor/     # 最小 Llama 模型层和分页 KV tensor
  entrypoints/        # API server 进程入口
  cache.py            # 物理块池和请求 block table
  scheduler.py        # v1 风格 token scheduler
runtime/
  minivllm_ops.cu     # Worker 调用的 CUDA C ABI
kernels/              # 原有 CUDA 算子与 benchmark
tests/                 # 内存、调度、多进程与异步并发测试
```

## 当前有意省略的部分

- tensor parallel / pipeline parallel；
- prefix caching、LoRA、speculative decoding；
- graph-safe 真实 Q/K/V 自定义 Paged Attention 的 CUDA Graph replay；
- Llama RoPE scaling、量化权重和非 Llama 模型；
- 分布式节点协调和生产级故障恢复；
- 完整 OpenAI API 与 vLLM 参数兼容层。

这些部分没有用空壳类伪装实现。系统中的扩展缝隙和推荐实现顺序记录在架构文档中。
