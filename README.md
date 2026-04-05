# miniVLLM: High-Performance LLM Inference Kernels

`miniVLLM` is a research-oriented library dedicated to implementing core LLM inference kernels from scratch. This project focuses on leveraging GPU memory hierarchy optimizations (Tiling & Register Reuse), Kernel Fusion, and advanced memory management techniques specifically designed for Large Language Models.

---

## Key Features & Optimization Evolution

### 1. Optimized General Matrix Multiplication (GeMMs)
A highly optimized GeMM implementation serves as the foundational building block for all linear layers.
* **Hierarchical Tiling**: Utilizes Shared Memory to cache data tiles, significantly reducing global memory transaction overhead.
* **Register Reuse**: Data is further promoted from Shared Memory to registers, maximizing the utilization of CUDA cores and reducing shared memory bank conflicts.
* **Thread Reusing**: Optimized thread mapping to ensure maximum hardware occupancy and instruction-level parallelism.

### 2. Flash Attention (Fused Kernel)
A high-performance attention mechanism designed to break the memory bottleneck of standard Transformers.
* **Kernel Fusion**: Fuses $QK^T$, Softmax, and $PV$ into a single kernel, preventing the materialization of the large $L \times L$ attention matrix in HBM.
* **Online Softmax with Warp Shuffle**: Implements the online update algorithm for maximums and normalization factors, enabling computation with $O(L)$ memory complexity. **Warp-level primitives (`__shfl_xor_sync`)** are utilized for ultra-fast horizontal reductions of local maximums and sums, bypassing Shared Memory latency.
* **SRAM Optimization**: Fine-grained tiling within Shared Memory to stay within the speed-of-light limits of the GPU.

### 3. Paged Attention
An advanced memory management system inspired by OS virtual memory to solve KV Cache fragmentation.
* **Non-contiguous KV Cache**: Supports storing Key and Value tensors in non-contiguous physical blocks (Paged Memory).
* **Dynamic Allocation**: Reduces VRAM fragmentation and enables higher effective throughput by allowing larger batch sizes for long-context inference.

---

## Performance Benchmarking

**Test Environment**: NVIDIA RTX 4060

### GFLOPS Comparison
| Kernel Type | Workload / Sequence Length | Naive (GFLOPS) | Optimized (GFLOPS) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **GeMM** | :--- | :--- | :--- | :--- |
| **Flash Attention** |:--- | :--- | :--- | :--- | :--- |

> 

### Profiler Analysis (NVIDIA Nsight Compute)
Detailed analysis using `ncu` (Nsight Compute) reveals the efficiency of our implementations:
* **Compute Throughput**: (on progress)
* **Memory Bandwidth**: (on progress)
* **Occupancy**: (on progress)

---

## Project Structure

```text
.
├── kernels/
│   ├── gemm/                # GeMM implementations (Tiling, Register/Thread Reuse)
│   ├── flash_attn/          # Fused Flash Attention & Online Softmax
│   └── paged_attn/          # Paged Attention memory management
├── bench/                   # GFLOPS benchmarking and validation scripts
├── include/                 # Shared headers, custom structs, and dimension utils
└── README.md
```

### Getting Started
Prerequisites
CUDA Toolkit (On progress)
CMake (On progress)

## Compilation
Bash
mkdir build && cd build
cmake ..
make -j$(nproc)
Running Benchmarks
Bash

### Benchmark Matrix Multiplication
./bench_gemm --m xxxx --n xxxx --k xxxx

### Benchmark Flash Attention
./bench_flash_attn --batch xx --heads xx --seq xxxx --dim xxx

### Acknowledgments
The GeMM implementation style is inspired by the NVIDIA CUDA Samples (matrixMul).
The Flash Attention implementation is based on the research by Tri Dao et al.
The Paged Attention mechanism follows the architecture pioneered by the vLLM team.

Developed by envymoon