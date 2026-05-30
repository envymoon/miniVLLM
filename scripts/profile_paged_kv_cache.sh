#!/usr/bin/env bash
set -euo pipefail

mkdir -p bench/bin bench/logs bench/reports

nvcc -O3 -arch=sm_89 -Iinclude --ptxas-options=-v \
  kernels/paged_kv_cache/paged_kv_cache.cu \
  -o bench/bin/paged_kv_cache \
  2>&1 | tee bench/logs/paged_kv_cache_build.log

bench/bin/paged_kv_cache "${1:-512}" "${2:-8}" \
  2>&1 | tee bench/logs/paged_kv_cache_run.txt

if command -v ncu >/dev/null 2>&1; then
  ncu --set full \
    --target-processes all \
    --export bench/reports/paged_kv_cache_report \
    --force-overwrite \
    bench/bin/paged_kv_cache "${1:-512}" "${2:-8}" \
    2>&1 | tee bench/logs/paged_kv_cache_ncu.txt
else
  echo "ncu not found; CUDA event timing saved in bench/logs/paged_kv_cache_profile.json"
fi
