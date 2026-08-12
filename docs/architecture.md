# miniVLLM 系统架构

## 1. 设计目标

本项目实现 vLLM v1 的核心请求链路：请求进入 EngineCore，Scheduler 以 token
为单位组成 batch，Executor/Worker 使用分页 KV Cache 执行模型，然后增量返回 token。

系统有三个约束：

1. EngineCore 是请求状态的唯一所有者，避免 API 线程和 Worker 同时修改状态；
2. KV 的逻辑分配发生在调度阶段，物理存储只存在于对应 DP Worker；
3. 算子层只接收 tensor/slot/block metadata，不理解 HTTP、请求队列或调度策略。

## 2. 组件与进程边界

```text
HTTP threads                         API process
    │
    ▼
AsyncLLMEngine ── EngineCoreClient   API process
    │ command queue     ▲ output queue
    ▼                   │
DP Coordinator (DP > 1)               Coordinator process
    ├─────────────────────────────┐
    ▼                             ▼
EngineCore + Scheduler rank 0  EngineCore + Scheduler rank 1
    │ WorkerBatch / Result         │ WorkerBatch / Result
    ▼                              ▼
ModelRunner rank 0             ModelRunner rank 1      Worker processes
KernelRuntime + KV cache       KernelRuntime + KV cache
```

### API / Facade

- `AsyncLLMEngine.generate()` 是异步生成器，不运行调度循环；
- `LLMEngine.generate()` 是同步生成器；
- 两者共用 `EngineCoreClient`，由一个 routing thread 将多请求输出分发到请求私有队列；
- API 客户端断开或生成器提前退出时发送 `ABORT`，EngineCore 释放 block table。

### EngineCore

每个 DP rank 有一个 EngineCore，并独占该 rank 的以下可变状态：

- waiting/running 请求集合；
- 每个请求的 `num_computed_tokens` 和 output token；
- rank-local KV block pool 与 block table；
- 全局 `max_num_batched_tokens` token budget。

EngineCore 不持有 CUDA tensor，也不执行 attention。

### DP Coordinator 与 Worker

`data_parallel_size=N` 会创建一个 DP Coordinator、N 个 EngineCore 和 N 个 Worker。
Coordinator 只维护 request-to-rank 映射与各 rank 的活动请求数，把新请求发给负载最小
的 EngineCore；它不参与 token 调度或 KV 分配。当前每个 rank 是完整模型副本的抽象，
每个 rank 独立推进请求，因此可以并发执行。每个 rank 的 block id 只在该 rank 内有意义。

## 3. 请求状态与 v1 token 调度

```text
WAITING
  │ Coordinator chooses rank; rank-local Scheduler admits
  ▼
RUNNING: chunked prefill
  │ num_computed_tokens == prompt tokens
  ▼
RUNNING: sample first output token
  │ append sampled token to all_token_ids
  ▼
RUNNING: one-token decode ─────┐
  │ sample next token          │
  └────────────────────────────┘
  │ EOS / stop / length / abort
  ▼
FINISHED: release all physical blocks
```

这里没有为 prefill 和 decode 建立两套请求对象。和 v1 思路一致，Scheduler 比较
`len(prompt_token_ids + output_token_ids)` 与 `num_computed_tokens`，差值就是尚未送进
模型的 token。长 prompt 会被 `max_num_batched_tokens` 自动切块；decode 通常差一个
token。

## 4. Paged KV Cache

每个 EngineCore 的 `KVCacheManager` 维护：

```text
logical token position
  -> logical_block = position // block_size
  -> physical_block = block_table[logical_block]
  -> slot = physical_block * block_size + position % block_size
```

Scheduler 在创建 `WorkerItem` 时产生 `slot_mapping`。Worker 使用它写 K/V，并在 decode
时用 `block_table + seq_len` 重新遍历历史 token。请求完成或 abort 后，所有物理块一次性
归还对应 rank 的 `BlockPool`。

当前版本不实现 swap 和 recompute preemption；单请求必须小于一个 rank 的 KV 容量。
这项限制会在请求进入 Scheduler 时明确报错，不会静默卡住。

## 5. Attention 接入

Worker 的固定调用顺序为：

```text
append_kv(token_ids, slot_mapping)
  ├─ generated_count == 0 -> flash_attention_prefill(prefix_slots)
  └─ generated_count > 0  -> paged_decode_attention(block_table, seq_len)
```

`ReferenceKernelRuntime` 用小向量执行相同的 online-softmax 与分页寻址，便于在没有 GPU
的机器验证系统。`CudaKernelRuntime` 通过 ctypes 加载 `runtime/minivllm_ops.cu` 提供的
C ABI。两者的接口完全相同。

原始优化 kernel 和 NCU 命令没有被塞进 Python 调度代码；优化 kernel 可以在稳定 ABI
后面替换 launch，再分别测量系统吞吐和 kernel 指标。

## 6. 源码阅读顺序

建议按一次请求的方向阅读：

1. `minivllm/engine/async_engine.py`：公开的流式接口；
2. `minivllm/engine/client.py`：API 与后端进程的通信；
3. `minivllm/engine/dp_coordinator.py`：请求到 DP rank 的负载均衡；
4. `minivllm/engine/core.py`：rank-local 事件循环与 Worker 执行；
5. `minivllm/scheduler.py`：token budget 与 chunked prefill；
6. `minivllm/cache.py`：block table 与 slot mapping；
7. `minivllm/worker/model_runner.py`：模型执行边界；
8. `minivllm/worker/operator_runtime.py`：CPU/CUDA 算子适配；
9. `runtime/minivllm_ops.cu` 与 `kernels/`：CUDA 实现和 NCU 分析。

## 7. 接入真实模型的推荐顺序

1. 增加模型配置和权重加载，只让 Worker 读取权重；
2. 将 `CharacterTokenizer` 换成模型 tokenizer，但仍在 API 进程编码；
3. 用真实 embedding、RMSNorm、GEMM、RoPE 和 LM head 替换
   `BootstrapModelRunner`；
4. 让 Q/K/V tensor 直接进入现有 Flash/Paged Attention ABI；
5. 增加采样器，使 temperature/top-p/top-k 真正作用于 logits；
6. 最后再增加 prefix caching、preemption 和跨节点通信。

这个顺序能始终保留一条可运行、可测试的请求链路，也便于定位错误属于系统层、模型层
还是算子层。
