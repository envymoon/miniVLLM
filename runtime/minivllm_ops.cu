// Minimal CUDA C ABI used by minivllm/worker/operator_runtime.py.
//
// The profiling kernels remain untouched in kernels/. This file only turns the
// same two system operations into a persistent Worker-owned runtime:
//   1. append token K/V vectors through a slot mapping;
//   2. attend through either dense prefill slots or a paged block table.

#include <cuda_runtime.h>

#include <cmath>
#include <new>

#if defined(_WIN32)
#define MVLLM_EXPORT extern "C" __declspec(dllexport)
#else
#define MVLLM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Runtime {
    int num_blocks;
    int block_size;
    int head_dim;
    float* key_cache;
    float* value_cache;
    float* attention_output;
    cudaStream_t stream;
};

__global__ void append_kernel(
    const int* token_ids,
    const int* slots,
    int count,
    int head_dim,
    float* key_cache,
    float* value_cache) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = count * head_dim;
    if (index >= total) {
        return;
    }
    const int token = index / head_dim;
    const int dim = index % head_dim;
    const int cache_index = slots[token] * head_dim + dim;
    key_cache[cache_index] = sinf(token_ids[token] * 0.01f + dim * 0.17f);
    value_cache[cache_index] = sinf(token_ids[token] * 0.01f + dim * 0.31f);
}

// A compact online-softmax reference launch. The optimized standalone kernels
// remain the NCU performance targets; this ABI prioritizes a correctness-first path.
__global__ void dense_last_token_attention_kernel(
    const float* key_cache,
    const float* value_cache,
    const int* slots,
    int count,
    int head_dim,
    float* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || count == 0) {
        return;
    }
    const int query_slot = slots[count - 1];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        output[dim] = 0.0f;
    }
    for (int token = 0; token < count; ++token) {
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += key_cache[query_slot * head_dim + dim] *
                     key_cache[slots[token] * head_dim + dim];
        }
        score *= scale;
        const float new_max = fmaxf(row_max, score);
        const float alpha = row_max == -INFINITY ? 0.0f : expf(row_max - new_max);
        const float beta = expf(score - new_max);
        for (int dim = 0; dim < head_dim; ++dim) {
            output[dim] = alpha * output[dim] +
                          beta * value_cache[slots[token] * head_dim + dim];
        }
        row_sum = alpha * row_sum + beta;
        row_max = new_max;
    }
    for (int dim = 0; dim < head_dim; ++dim) {
        output[dim] /= row_sum;
    }
}

__global__ void paged_last_token_attention_kernel(
    const float* key_cache,
    const float* value_cache,
    const int* block_table,
    int seq_len,
    int block_size,
    int head_dim,
    float* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || seq_len == 0) {
        return;
    }
    const int query_position = seq_len - 1;
    const int query_slot = block_table[query_position / block_size] * block_size +
                           query_position % block_size;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        output[dim] = 0.0f;
    }
    for (int token = 0; token < seq_len; ++token) {
        const int slot = block_table[token / block_size] * block_size +
                         token % block_size;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += key_cache[query_slot * head_dim + dim] *
                     key_cache[slot * head_dim + dim];
        }
        score *= scale;
        const float new_max = fmaxf(row_max, score);
        const float alpha = row_max == -INFINITY ? 0.0f : expf(row_max - new_max);
        const float beta = expf(score - new_max);
        for (int dim = 0; dim < head_dim; ++dim) {
            output[dim] = alpha * output[dim] +
                          beta * value_cache[slot * head_dim + dim];
        }
        row_sum = alpha * row_sum + beta;
        row_max = new_max;
    }
    for (int dim = 0; dim < head_dim; ++dim) {
        output[dim] /= row_sum;
    }
}

int copy_ints(int** device, const int* host, int count, cudaStream_t stream) {
    if (count == 0) {
        *device = nullptr;
        return 0;
    }
    if (cudaMalloc(device, static_cast<size_t>(count) * sizeof(int)) != cudaSuccess) {
        return 1;
    }
    if (cudaMemcpyAsync(
            *device,
            host,
            static_cast<size_t>(count) * sizeof(int),
            cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        cudaFree(*device);
        *device = nullptr;
        return 2;
    }
    return 0;
}

}  // namespace

MVLLM_EXPORT void* mvllm_create(
    int device_index, int num_blocks, int block_size, int head_dim) {
    if (device_index < 0 || num_blocks <= 0 || block_size <= 0 || head_dim <= 0) {
        return nullptr;
    }
    if (cudaSetDevice(device_index) != cudaSuccess) {
        return nullptr;
    }
    Runtime* runtime = new (std::nothrow) Runtime{};
    if (runtime == nullptr) {
        return nullptr;
    }
    runtime->num_blocks = num_blocks;
    runtime->block_size = block_size;
    runtime->head_dim = head_dim;
    const size_t elements = static_cast<size_t>(num_blocks) * block_size * head_dim;
    if (cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&runtime->key_cache, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&runtime->value_cache, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&runtime->attention_output, head_dim * sizeof(float)) != cudaSuccess) {
        if (runtime->key_cache) cudaFree(runtime->key_cache);
        if (runtime->value_cache) cudaFree(runtime->value_cache);
        if (runtime->attention_output) cudaFree(runtime->attention_output);
        if (runtime->stream) cudaStreamDestroy(runtime->stream);
        delete runtime;
        return nullptr;
    }
    cudaMemset(runtime->key_cache, 0, elements * sizeof(float));
    cudaMemset(runtime->value_cache, 0, elements * sizeof(float));
    return runtime;
}

MVLLM_EXPORT void mvllm_destroy(void* opaque) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr) return;
    cudaStreamSynchronize(runtime->stream);
    cudaFree(runtime->key_cache);
    cudaFree(runtime->value_cache);
    cudaFree(runtime->attention_output);
    cudaStreamDestroy(runtime->stream);
    delete runtime;
}

MVLLM_EXPORT int mvllm_append(
    void* opaque, const int* token_ids, const int* slots, int count) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr || count < 0) return 10;
    int *d_tokens = nullptr, *d_slots = nullptr;
    if (copy_ints(&d_tokens, token_ids, count, runtime->stream) != 0) return 11;
    if (copy_ints(&d_slots, slots, count, runtime->stream) != 0) {
        cudaFree(d_tokens);
        return 11;
    }
    const int threads = 256;
    const int total = count * runtime->head_dim;
    if (total > 0) {
        append_kernel<<<(total + threads - 1) / threads, threads, 0, runtime->stream>>>(
            d_tokens, d_slots, count, runtime->head_dim,
            runtime->key_cache, runtime->value_cache);
    }
    const cudaError_t status = cudaStreamSynchronize(runtime->stream);
    cudaFree(d_tokens);
    cudaFree(d_slots);
    return status == cudaSuccess ? 0 : 12;
}

MVLLM_EXPORT int mvllm_prefill(void* opaque, const int* slots, int count) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr || count < 0) return 20;
    int* d_slots = nullptr;
    if (copy_ints(&d_slots, slots, count, runtime->stream) != 0) return 21;
    dense_last_token_attention_kernel<<<1, 1, 0, runtime->stream>>>(
        runtime->key_cache, runtime->value_cache, d_slots, count,
        runtime->head_dim, runtime->attention_output);
    const cudaError_t status = cudaStreamSynchronize(runtime->stream);
    cudaFree(d_slots);
    return status == cudaSuccess ? 0 : 22;
}

MVLLM_EXPORT int mvllm_decode(
    void* opaque, const int* block_table, int num_table_blocks, int seq_len) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr || num_table_blocks < 0 || seq_len < 0) return 30;
    int* d_table = nullptr;
    if (copy_ints(&d_table, block_table, num_table_blocks, runtime->stream) != 0) {
        return 31;
    }
    paged_last_token_attention_kernel<<<1, 1, 0, runtime->stream>>>(
        runtime->key_cache, runtime->value_cache, d_table, seq_len,
        runtime->block_size, runtime->head_dim, runtime->attention_output);
    const cudaError_t status = cudaStreamSynchronize(runtime->stream);
    cudaFree(d_table);
    return status == cudaSuccess ? 0 : 32;
}
