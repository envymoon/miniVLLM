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
GPUInputBatch + KV cache       GPUInputBatch + KV cache
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

Scheduler 允许新进入的 prefill 请求和已经运行的 decode 请求出现在同一轮。它输出的
`WorkerItem` 列表在 Worker 中压平为一个 `GPUInputBatch`：

```text
input_ids / positions / slot_mapping        flat token axis
query_start_loc                             request -> query range
seq_lens / block_tables                     request -> paged context
context_slot_mapping / context_start_loc    reference attention metadata
sample_indices                              request -> logits row
```

Worker 每轮只调用一次 `ModelRunner.execute_model(batch)`。默认 runtime 也只执行一次
批量 KV append 和一次批量 attention 入口，不再按请求调用 ModelRunner。

每个 batch 会报告 scheduled/prefill/decode token 数、请求数、实际执行模式和 Worker
执行时间；请求状态另外记录 TTFT、最近一次 ITL 与端到端延迟。这些字段位于公开输出
协议中，可直接作为后续调度策略实验的原始数据。

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

默认 bootstrap Worker 的固定调用顺序为：

```text
append_kv(flat_input_ids, flat_slot_mapping)
  └─ attention_batch(
       context_slot_mapping,
       context_start_loc,
       block_tables,
       seq_lens,
       is_decode)
```

`ReferenceKernelRuntime` 用小向量执行相同的 online-softmax 与分页寻址，便于在没有 GPU
的机器验证系统。`CudaKernelRuntime` 通过 ctypes 加载 `runtime/minivllm_ops.cu` 提供的
C ABI。两者的接口完全相同。

配置本地模型目录时，`LlamaModelRunner` 加载标准 safetensors 权重。每层维护独立的
`[num_blocks * block_size, num_kv_heads, head_dim]` K/V tensor；Q/K 使用 RoPE 后，K/V
通过 `slot_mapping` 写入物理位置，attention 再通过请求 block table 对应的 context slots
读取。prefill 与 decode 可以在同一次模型调用中执行。

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
7. `minivllm/worker/batch.py`：调度输出到 GPU batch metadata；
8. `minivllm/worker/model_runner.py`：模型执行边界；
9. `minivllm/model_executor/llama.py`：最小 Llama 与层级分页 KV Cache；
10. `minivllm/worker/cudagraph.py`：decode graph 分派和 eager fallback；
11. `minivllm/worker/operator_runtime.py`：CPU/CUDA 算子适配；
12. `runtime/minivllm_ops.cu` 与 `kernels/`：CUDA 实现和 NCU 分析。

## 7. CUDA Graph 边界

`CUDAGraphDispatcher` 只把 uniform decode batch 映射到配置的 capture bucket。mixed
prefill/decode 和超过最大 bucket 的 batch 都回退 eager。Manager 只有在 ModelRunner
明确提供 `execute_cudagraph(batch, capture_size)` 时才会报告 `cuda_graph`。CUDA
bootstrap runtime 为每个 capture size 缓存一个 `cudaGraphExec_t`，其 graph 包含批量
KV append 与 paged decode attention；每次 replay 前只更新持久 device input buffer。
当前 Llama 路径仍是动态 gather，因此不会虚报 graph replay。

真实模型 capture/replay 的剩余工作集中在 Worker 内：让层级 Q/K/V 使用 graph-safe
paged decode kernel，并复用已经存在的静态 buffer、dummy padding 和 capture cache
模式。Scheduler 与 IPC 协议无需因 CUDA Graph 改变。

## 8. 后续实现顺序

1. 让真实 Q/K/V tensor 直接进入定制 Flash/Paged Attention ABI；
2. 为纯 decode 实现静态输入 buffer 和 CUDA Graph capture/replay；
3. 增加 top-p/top-k sampler 和确定性随机数状态；
4. 建立 workload generator、TTFT/ITL/JCT 与公平性实验指标；
5. 接入可插拔调度策略、prefix caching、preemption 和跨节点通信。
