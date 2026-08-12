// Minimal CUDA C ABI used by minivllm/worker/operator_runtime.py.
//
// The profiling kernels remain untouched in kernels/. This file only turns the
// same two system operations into a persistent Worker-owned runtime:
//   1. append token K/V vectors through a slot mapping;
//   2. attend through either dense prefill slots or a paged block table.

#include <cuda_runtime.h>

#include <cmath>
#include <new>
#include <unordered_map>
#include <vector>

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
    int graph_capacity_requests;
    int graph_max_blocks;
    int physical_cache_slots;
    float* key_cache;
    float* value_cache;
    float* attention_output;
    int* graph_token_ids;
    int* graph_slots;
    int* graph_block_tables;
    int* graph_seq_lens;
    int* graph_is_decode;
    int* graph_context_slots;
    int* graph_context_start_locs;
    std::unordered_map<int, cudaGraphExec_t> decode_graphs;
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

// One grid launch processes every request in a scheduler batch. Prefill rows use
// the dense context slot list; decode rows walk their paged block table.
__global__ void batched_last_token_attention_kernel(
    const float* key_cache,
    const float* value_cache,
    const int* context_slots,
    const int* context_start_locs,
    const int* block_tables,
    int max_blocks,
    const int* seq_lens,
    const int* is_decode,
    int num_requests,
    int block_size,
    int head_dim,
    float* outputs) {
    const int request = blockIdx.x;
    if (request >= num_requests || threadIdx.x != 0) {
        return;
    }
    const int seq_len = seq_lens[request];
    if (seq_len <= 0) {
        return;
    }
    const int dense_start = context_start_locs[request];
    const int query_position = seq_len - 1;
    const int query_slot = is_decode[request]
        ? block_tables[request * max_blocks + query_position / block_size] * block_size
              + query_position % block_size
        : context_slots[dense_start + query_position];
    float* output = outputs + static_cast<size_t>(request) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        output[dim] = 0.0f;
    }
    for (int token = 0; token < seq_len; ++token) {
        const int slot = is_decode[request]
            ? block_tables[request * max_blocks + token / block_size] * block_size
                  + token % block_size
            : context_slots[dense_start + token];
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

int get_or_create_decode_graph(Runtime* runtime, int capture_size) {
    if (runtime->decode_graphs.find(capture_size) != runtime->decode_graphs.end()) {
        return 0;
    }
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(runtime->stream, cudaStreamCaptureModeGlobal) !=
        cudaSuccess) {
        return 1;
    }
    const int threads = 256;
    const int total = capture_size * runtime->head_dim;
    append_kernel<<<(total + threads - 1) / threads, threads, 0, runtime->stream>>>(
        runtime->graph_token_ids,
        runtime->graph_slots,
        capture_size,
        runtime->head_dim,
        runtime->key_cache,
        runtime->value_cache);
    batched_last_token_attention_kernel<<<capture_size, 1, 0, runtime->stream>>>(
        runtime->key_cache,
        runtime->value_cache,
        runtime->graph_context_slots,
        runtime->graph_context_start_locs,
        runtime->graph_block_tables,
        runtime->graph_max_blocks,
        runtime->graph_seq_lens,
        runtime->graph_is_decode,
        capture_size,
        runtime->block_size,
        runtime->head_dim,
        runtime->attention_output);
    if (cudaStreamEndCapture(runtime->stream, &graph) != cudaSuccess) {
        return 2;
    }
    cudaGraphExec_t executable = nullptr;
    const cudaError_t instantiate_status =
        cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (instantiate_status != cudaSuccess) {
        return 3;
    }
    runtime->decode_graphs.emplace(capture_size, executable);
    return 0;
}

}  // namespace

MVLLM_EXPORT void* mvllm_create(
    int device_index,
    int num_blocks,
    int block_size,
    int head_dim,
    int max_num_seqs,
    int max_model_len) {
    if (device_index < 0 || num_blocks <= 0 || block_size <= 0 || head_dim <= 0 ||
        max_num_seqs <= 0 || max_model_len <= 0) {
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
    runtime->graph_capacity_requests = max_num_seqs;
    runtime->graph_max_blocks = (max_model_len + block_size - 1) / block_size;
    runtime->physical_cache_slots = num_blocks * block_size;
    // Scratch slots make padded graph rows independent from live request slots.
    const size_t cache_slots =
        static_cast<size_t>(runtime->physical_cache_slots) + max_num_seqs;
    const size_t elements = cache_slots * head_dim;
    if (cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&runtime->key_cache, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&runtime->value_cache, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(
            &runtime->attention_output,
            static_cast<size_t>(max_num_seqs) * head_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_token_ids, max_num_seqs * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&runtime->graph_slots, max_num_seqs * sizeof(int)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_block_tables,
            static_cast<size_t>(max_num_seqs) * runtime->graph_max_blocks *
                sizeof(int)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_seq_lens, max_num_seqs * sizeof(int)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_is_decode, max_num_seqs * sizeof(int)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_context_slots, max_num_seqs * sizeof(int)) != cudaSuccess ||
        cudaMalloc(
            &runtime->graph_context_start_locs,
            static_cast<size_t>(max_num_seqs + 1) * sizeof(int)) != cudaSuccess) {
        if (runtime->key_cache) cudaFree(runtime->key_cache);
        if (runtime->value_cache) cudaFree(runtime->value_cache);
        if (runtime->attention_output) cudaFree(runtime->attention_output);
        if (runtime->graph_token_ids) cudaFree(runtime->graph_token_ids);
        if (runtime->graph_slots) cudaFree(runtime->graph_slots);
        if (runtime->graph_block_tables) cudaFree(runtime->graph_block_tables);
        if (runtime->graph_seq_lens) cudaFree(runtime->graph_seq_lens);
        if (runtime->graph_is_decode) cudaFree(runtime->graph_is_decode);
        if (runtime->graph_context_slots) cudaFree(runtime->graph_context_slots);
        if (runtime->graph_context_start_locs) {
            cudaFree(runtime->graph_context_start_locs);
        }
        if (runtime->stream) cudaStreamDestroy(runtime->stream);
        delete runtime;
        return nullptr;
    }
    cudaMemset(runtime->key_cache, 0, elements * sizeof(float));
    cudaMemset(runtime->value_cache, 0, elements * sizeof(float));
    cudaMemset(runtime->graph_is_decode, 1, max_num_seqs * sizeof(int));
    cudaMemset(runtime->graph_context_slots, 0, max_num_seqs * sizeof(int));
    cudaMemset(
        runtime->graph_context_start_locs,
        0,
        static_cast<size_t>(max_num_seqs + 1) * sizeof(int));
    return runtime;
}

MVLLM_EXPORT void mvllm_destroy(void* opaque) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr) return;
    cudaStreamSynchronize(runtime->stream);
    for (const auto& item : runtime->decode_graphs) {
        cudaGraphExecDestroy(item.second);
    }
    cudaFree(runtime->key_cache);
    cudaFree(runtime->value_cache);
    cudaFree(runtime->attention_output);
    cudaFree(runtime->graph_token_ids);
    cudaFree(runtime->graph_slots);
    cudaFree(runtime->graph_block_tables);
    cudaFree(runtime->graph_seq_lens);
    cudaFree(runtime->graph_is_decode);
    cudaFree(runtime->graph_context_slots);
    cudaFree(runtime->graph_context_start_locs);
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

MVLLM_EXPORT int mvllm_attention_batch(
    void* opaque,
    const int* context_slots,
    const int* context_start_locs,
    const int* block_tables,
    int max_blocks,
    const int* seq_lens,
    const int* is_decode,
    int num_requests) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr || num_requests <= 0 ||
        num_requests > runtime->num_blocks ||
        num_requests > runtime->graph_capacity_requests || max_blocks <= 0) {
        return 40;
    }
    const int context_count = context_start_locs[num_requests];
    int *d_context_slots = nullptr, *d_context_start_locs = nullptr;
    int *d_block_tables = nullptr, *d_seq_lens = nullptr, *d_is_decode = nullptr;
    if (copy_ints(&d_context_slots, context_slots, context_count, runtime->stream) != 0 ||
        copy_ints(
            &d_context_start_locs,
            context_start_locs,
            num_requests + 1,
            runtime->stream) != 0 ||
        copy_ints(
            &d_block_tables,
            block_tables,
            num_requests * max_blocks,
            runtime->stream) != 0 ||
        copy_ints(&d_seq_lens, seq_lens, num_requests, runtime->stream) != 0 ||
        copy_ints(&d_is_decode, is_decode, num_requests, runtime->stream) != 0) {
        cudaFree(d_context_slots);
        cudaFree(d_context_start_locs);
        cudaFree(d_block_tables);
        cudaFree(d_seq_lens);
        cudaFree(d_is_decode);
        return 41;
    }
    batched_last_token_attention_kernel<<<num_requests, 1, 0, runtime->stream>>>(
        runtime->key_cache,
        runtime->value_cache,
        d_context_slots,
        d_context_start_locs,
        d_block_tables,
        max_blocks,
        d_seq_lens,
        d_is_decode,
        num_requests,
        runtime->block_size,
        runtime->head_dim,
        runtime->attention_output);
    const cudaError_t status = cudaStreamSynchronize(runtime->stream);
    cudaFree(d_context_slots);
    cudaFree(d_context_start_locs);
    cudaFree(d_block_tables);
    cudaFree(d_seq_lens);
    cudaFree(d_is_decode);
    return status == cudaSuccess ? 0 : 42;
}

MVLLM_EXPORT int mvllm_execute_decode_graph(
    void* opaque,
    const int* token_ids,
    const int* slots,
    const int* block_tables,
    int input_max_blocks,
    const int* seq_lens,
    int num_requests,
    int capture_size) {
    Runtime* runtime = static_cast<Runtime*>(opaque);
    if (runtime == nullptr || num_requests <= 0 || capture_size < num_requests ||
        capture_size > runtime->graph_capacity_requests || input_max_blocks <= 0 ||
        input_max_blocks > runtime->graph_max_blocks) {
        return 50;
    }

    std::vector<int> padded_tokens(capture_size);
    std::vector<int> padded_slots(capture_size);
    std::vector<int> padded_seq_lens(capture_size);
    std::vector<int> padded_tables(
        static_cast<size_t>(capture_size) * runtime->graph_max_blocks, -1);
    for (int request = 0; request < capture_size; ++request) {
        const int source = request < num_requests ? request : num_requests - 1;
        padded_tokens[request] = token_ids[source];
        padded_slots[request] = request < num_requests
            ? slots[source]
            : runtime->physical_cache_slots + request;
        padded_seq_lens[request] = seq_lens[source];
        for (int block = 0; block < input_max_blocks; ++block) {
            padded_tables[
                static_cast<size_t>(request) * runtime->graph_max_blocks + block] =
                block_tables[source * input_max_blocks + block];
        }
    }

    if (cudaMemcpyAsync(
            runtime->graph_token_ids,
            padded_tokens.data(),
            static_cast<size_t>(capture_size) * sizeof(int),
            cudaMemcpyHostToDevice,
            runtime->stream) != cudaSuccess ||
        cudaMemcpyAsync(
            runtime->graph_slots,
            padded_slots.data(),
            static_cast<size_t>(capture_size) * sizeof(int),
            cudaMemcpyHostToDevice,
            runtime->stream) != cudaSuccess ||
        cudaMemcpyAsync(
            runtime->graph_seq_lens,
            padded_seq_lens.data(),
            static_cast<size_t>(capture_size) * sizeof(int),
            cudaMemcpyHostToDevice,
            runtime->stream) != cudaSuccess ||
        cudaMemcpyAsync(
            runtime->graph_block_tables,
            padded_tables.data(),
            static_cast<size_t>(capture_size) * runtime->graph_max_blocks * sizeof(int),
            cudaMemcpyHostToDevice,
            runtime->stream) != cudaSuccess) {
        return 51;
    }
    if (cudaStreamSynchronize(runtime->stream) != cudaSuccess) {
        return 52;
    }
    if (get_or_create_decode_graph(runtime, capture_size) != 0) {
        return 53;
    }
    const cudaError_t launch_status = cudaGraphLaunch(
        runtime->decode_graphs.at(capture_size), runtime->stream);
    if (launch_status != cudaSuccess ||
        cudaStreamSynchronize(runtime->stream) != cudaSuccess) {
        return 54;
    }
    return 0;
}
