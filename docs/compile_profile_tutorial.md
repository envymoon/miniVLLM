# miniVLLM Compile / Run / Profile Workflow

This tutorial focuses only on the practical workflow:

1. compile the kernel source files
2. run the executables
3. profile them with Nsight Compute
4. record the key data in JSON

It assumes you already have a working CUDA build environment.

---

## 1. Move to the Project Root

If you are using WSL:

```bash
cd /mnt/c/Users/ianha/Downloads/Projects/miniVLLM
```

Create the output folders once:

```bash
mkdir -p bench/bin bench/reports bench/logs
```

---

## 2. Compile `GeMMs.cu`

Example command:

```bash
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 \
  -I /path/to/cuda-samples/common/inc \
  kernels/GeMMs.cu \
  -o bench/bin/gemms
```

If your helper headers are already in a usable include path, you can drop the `-I`.

What this does:

- `-O3`: optimized build
- `-lineinfo`: keeps source line mapping for profiler reports
- `-std=c++17`: modern C++ mode
- `-arch=sm_89`: Ada generation target, suitable for RTX 4060

---

## 3. Run `GeMMs`

```bash
./bench/bin/gemms | tee bench/logs/gemms_run.txt
```

Read the printed output and record:

- elapsed time
- GFLOPS
- whether the executable finishes cleanly

---

## 4. Profile `GeMMs`

Fast CSV pass:

```bash
ncu \
  --target-processes all \
  --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed.avg,\
    smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
    dram__bytes.sum \
  --csv \
  ./bench/bin/gemms | tee bench/logs/gemms_ncu.csv
```

Full report export:

```bash
ncu \
  --set full \
  --target-processes all \
  --export bench/reports/gemms_full \
  --force-overwrite \
  ./bench/bin/gemms
```

Key fields to read:

- occupancy
- registers per thread
- shared memory per block
- DRAM bytes
- instruction count
- FMA pipe active percent

---

## 5. Compile `flash_attention.cu`

Example command:

```bash
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 \
  -I /path/to/cuda-samples/common/inc \
  kernels/flash_attention.cu \
  -o bench/bin/flash_attention
```

---

## 6. Run `flash_attention`

```bash
./bench/bin/flash_attention | tee bench/logs/flash_attention_run.txt
```

The current source already prints:

- validation header
- max absolute error
- max relative error
- NaN / Inf flag
- average runtime
- estimated TFLOPs

These are the first values you should record before trusting profiler data.

---

## 7. Profile `flash_attention`

Fast CSV pass:

```bash
ncu \
  --target-processes all \
  --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed.avg,\
    smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
    dram__bytes.sum \
  --csv \
  ./bench/bin/flash_attention | tee bench/logs/flash_attention_ncu.csv
```

Full report export:

```bash
ncu \
  --set full \
  --target-processes all \
  --export bench/reports/flash_attention_full \
  --force-overwrite \
  ./bench/bin/flash_attention
```

Key fields to read:

- occupancy
- registers per thread
- shared memory per block
- DRAM bytes
- instruction count
- warp activity

If `ncu` prints:

```text
ERR_NVGPUCTRPERM
```

that means the executable launched correctly, but the current user does not have permission to read GPU performance counters. In that case:

- runtime benchmarking is still valid
- correctness checks are still valid
- profiler metrics will remain unavailable until counter access is enabled on the system

---

## 8. Record the Results in JSON

The project includes:

```text
bench/profile_results.json
```

After each run, fill in:

- correctness
- timing
- NCU metrics
- notes about whether the configuration was stable

Suggested workflow:

1. run the executable
2. copy the printed timing and correctness data into JSON
3. run `ncu`
4. copy the important metrics into JSON
5. repeat with different `block_size`, `head_dim`, or matrix sizes

If profiler access is blocked, still record:

- compile succeeded or not
- runtime output
- validation output
- the exact profiler error string

---

## 9. Recommended Reading Order

For learning, use this order:

1. compile and run `GeMMs`
2. inspect the profiler report for `GeMMs`
3. only after that, move to `flash_attention`
4. first verify correctness for `flash_attention`
5. then read the profiler report

This order is easier because GeMM is simpler to reason about than fused attention.

---

## 10. Minimal Command Sequence

If your environment is already correct, this is the shortest useful workflow:

```bash
cd /mnt/c/Users/ianha/Downloads/Projects/miniVLLM
mkdir -p bench/bin bench/reports bench/logs

nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -I /path/to/cuda-samples/common/inc kernels/GeMMs.cu -o bench/bin/gemms
./bench/bin/gemms | tee bench/logs/gemms_run.txt
ncu --set full --target-processes all --export bench/reports/gemms_full --force-overwrite ./bench/bin/gemms

nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -I /path/to/cuda-samples/common/inc kernels/flash_attention.cu -o bench/bin/flash_attention
./bench/bin/flash_attention | tee bench/logs/flash_attention_run.txt
ncu --set full --target-processes all --export bench/reports/flash_attention_full --force-overwrite ./bench/bin/flash_attention
```

Then update:

```text
bench/profile_results.json
```
