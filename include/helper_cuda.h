#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

inline void __check_cuda_errors(cudaError_t err,
                                const char* expr,
                                const char* file,
                                int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error at %s:%d for %s: %s\n",
                     file,
                     line,
                     expr,
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) __check_cuda_errors((val), #val, __FILE__, __LINE__)
#define cudaCheckErrors(val) __check_cuda_errors((val), #val, __FILE__, __LINE__)
