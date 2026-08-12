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
- prefill 进入 Flash Attention 接口，decode 通过 block table 进入 Paged Attention；
- 原有 `kernels/`、`bench/` 和 NCU 脚本保持为算子性能分析入口。

> 仓库目前没有 Transformer 权重、模型层或真实 tokenizer。默认
> `reference` Worker 当前使用确定性的 bootstrap ModelRunner，用于系统验证，输出不代表
> 真实 LLM。接入模型时只需替换 `BootstrapModelRunner`，无需改动调度和 IPC。

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

项目的系统层没有第三方 Python 依赖。

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

## CUDA 算子运行时

原有算子 benchmark 不变：

```bash
make
./scripts/profile_paged_kv_cache.sh 512 8
```

另外提供了 Worker 可加载的稳定 C ABI。它持久持有 KV Cache，并接受 EngineCore 产生
的 slot mapping 和 block table：

```bash
make native
python -m minivllm.entrypoints.api_server --backend cuda
```

也可用 `MINIVLLM_OPS_LIBRARY` 指向生成的 `.so`、`.dll` 或 `.dylib`。当前 C ABI 是
correctness-first 的参考 launch；`kernels/flash_attention/flash_attention_prefilling.cu` 和
`kernels/paged_kv_cache/paged_kv_cache.cu` 仍是优化与 NCU 的主战场。把优化 kernel
替换进 C ABI 不会影响系统层。

## 目录

```text
minivllm/
  engine/             # facade、DP Coordinator、EngineCore、IPC client
  worker/             # Worker 进程、ModelRunner、算子适配层
  entrypoints/        # API server 进程入口
  cache.py            # 物理块池和请求 block table
  scheduler.py        # v1 风格 token scheduler
runtime/
  minivllm_ops.cu     # Worker 调用的 CUDA C ABI
kernels/              # 原有 CUDA 算子与 benchmark
tests/                 # 内存、调度、多进程与异步并发测试
```

## 当前有意省略的部分

- 模型权重加载、Transformer layers 和真实 logits；
- tensor parallel / pipeline parallel；
- prefix caching、LoRA、speculative decoding；
- 分布式节点协调和生产级故障恢复；
- 完整 OpenAI API 与 vLLM 参数兼容层。

这些部分没有用空壳类伪装实现。系统中的扩展缝隙和推荐实现顺序记录在架构文档中。
