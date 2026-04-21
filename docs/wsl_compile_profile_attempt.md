# WSL Compile / Profile Attempt Record

Date: 2026-04-19

This file records the actual attempt to compile and profile the current kernels inside WSL during this session.

## What was confirmed

The project root is reachable from WSL:

```bash
/mnt/c/Users/ianha/Downloads/Projects/miniVLLM
```

The repository contains:

- `kernels/GeMMs.cu`
- `kernels/flash_attention.cu`

GPU visibility inside WSL works:

```bash
nvidia-smi
```

This successfully reported the RTX 4060.

## What was completed

### GeMMs

Compiled successfully with:

```bash
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -I include kernels/GeMMs.cu -o bench/bin/gemms
```

Runtime output:

```text
Performance= 45.47 GFlop/s, Time= 0.738 msec, Size= 33554432 Ops, WorkgroupSize= 64 threads/block
```

### Flash Attention

Compiled successfully with:

```bash
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -I include kernels/flash_attention.cu -o bench/bin/flash_attention
```

Runtime output:

```text
flash attention validation
B=1 H=2 S=32 D=64 block=32
max_abs_error=0.00000021 max_rel_error=0.00210040 has_nan=false
avg_time_ms=0.0955 estimated_TFLOPs=0.0055
```

## What blocked the profiler metrics

The CUDA compiler was still not available:

```bash
nvcc --version
```

Result:

- `nvcc: command not found`

No Linux Nsight Compute binary was found either:

```bash
which ncu
which nsys
```

Result:

- no path returned in this session

## Practical meaning

This means the session completed:

1. CUDA compilation in WSL
2. kernel execution from WSL binaries

but could not complete:

3. Nsight Compute metric collection

## Next step

Once GPU performance counter permissions are enabled, use the commands documented in:

```text
docs/compile_profile_tutorial.md
```

Then store the measured numbers in:

```text
bench/profile_results.json
```
