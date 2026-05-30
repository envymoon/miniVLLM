// ============================================================================
// Paged KV Cache + Paged Decode Attention
// ============================================================================
// This file is intentionally standalone for learning and profiling.
//
// Implemented pieces:
// 1. host-side block allocator with sequence block tables
// 2. append/copy kernel for writing K/V tokens into paged physical blocks
// 3. decode attention kernel that reads K/V through block tables
// 4. CPU reference validation and JSON profiling output
//
// Layout:
// - k_cache/v_cache: [num_blocks, block_size, kv_heads, head_dim]
// - key/value input: [total_tokens, kv_heads, head_dim]
// - query/output: [batch, query_heads, head_dim]
// - block_tables: [batch, max_blocks_per_seq]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                         \
    do {                                                                        \
        cudaError_t status = (expr);                                            \
        if (status != cudaSuccess) {                                            \
            fprintf(stderr,                                                     \
                    "CUDA error %s:%d: %s\n",                                  \
                    __FILE__,                                                   \
                    __LINE__,                                                   \
                    cudaGetErrorString(status));                                \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int kThreads = 128;
constexpr int kWarpSize = 32;

struct Shape {
    int batch_size = 8;
    int query_heads = 8;
    int kv_heads = 4;
    int head_dim = 64;
    int block_size = 16;
    int max_seq_len = 512;
    int max_blocks_per_seq = 64;
};

struct SequenceBlocks {
    int length = 0;
    std::vector<int> blocks;
};

class BlockAllocator {
public:
    explicit BlockAllocator(int total_blocks) {
        if (total_blocks <= 0) {
            throw std::invalid_argument("total_blocks must be positive");
        }
        free_blocks_.reserve(total_blocks);
        for (int block = total_blocks - 1; block >= 0; --block) {
            free_blocks_.push_back(block);
        }
    }

    int allocate() {
        if (free_blocks_.empty()) {
            throw std::runtime_error("paged KV cache OOM: no free physical blocks");
        }
        int block = free_blocks_.back();
        free_blocks_.pop_back();
        return block;
    }

    int available() const {
        return static_cast<int>(free_blocks_.size());
    }

private:
    std::vector<int> free_blocks_;
};

float rand_uniform(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    return dist(rng);
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void append_kv_cache_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int* __restrict__ token_offsets,
    const int* __restrict__ block_tables,
    const int seq_idx,
    const int max_blocks_per_seq,
    const int block_size,
    const int kv_heads,
    const int head_dim,
    const int total_tokens,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache) {

    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int elems_per_token = kv_heads * head_dim;
    const int total_elems = total_tokens * elems_per_token;
    if (linear >= total_elems) {
        return;
    }

    const int token_idx = linear / elems_per_token;
    const int within = linear % elems_per_token;
    const int kv_head = within / head_dim;
    const int dim = within % head_dim;

    const int absolute_pos = token_offsets[token_idx];
    const int logical_block = absolute_pos / block_size;
    const int block_offset = absolute_pos % block_size;
    const int physical_block =
        block_tables[seq_idx * max_blocks_per_seq + logical_block];

    const int cache_idx =
        ((physical_block * block_size + block_offset) * kv_heads + kv_head) *
            head_dim +
        dim;
    k_cache[cache_idx] = key[linear];
    v_cache[cache_idx] = value[linear];
}

__global__ void paged_decode_attention_kernel(
    const half* __restrict__ query,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int batch_size,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    half* __restrict__ output) {

    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;

    if (seq_idx >= batch_size || q_head >= query_heads) {
        return;
    }

    const int q_per_kv = query_heads / kv_heads;
    const int kv_head = q_head / q_per_kv;
    const int seq_len = seq_lens[seq_idx];
    const float scale = rsqrtf(static_cast<float>(head_dim));

    extern __shared__ float shared[];
    float* out_accum = shared;
    float* warp_out = shared + head_dim;
    float* warp_sum_buf = warp_out + warps * head_dim;
    float* warp_max_buf = warp_sum_buf + warps;
    float* global_stats = warp_max_buf + warps;

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        out_accum[dim] = 0.0f;
    }
    if (threadIdx.x < warps) {
        warp_sum_buf[threadIdx.x] = 0.0f;
        warp_max_buf[threadIdx.x] = -INFINITY;
    }
    for (int idx = threadIdx.x; idx < warps * head_dim; idx += blockDim.x) {
        warp_out[idx] = 0.0f;
    }
    __syncthreads();

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int token = warp_id; token < seq_len; token += warps) {
        const int logical_block = token / block_size;
        const int block_offset = token % block_size;
        const int physical_block =
            block_tables[seq_idx * max_blocks_per_seq + logical_block];

        float score = 0.0f;
        for (int dim = lane; dim < head_dim; dim += kWarpSize) {
            const int q_idx = (seq_idx * query_heads + q_head) * head_dim + dim;
            const int k_idx =
                ((physical_block * block_size + block_offset) * kv_heads + kv_head) *
                    head_dim +
                dim;
            score += __half2float(query[q_idx]) * __half2float(k_cache[k_idx]);
        }
        score = warp_sum(score);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float new_max = fmaxf(local_max, score);
        const float alpha =
            (local_max == -INFINITY) ? 0.0f : expf(local_max - new_max);
        const float beta = expf(score - new_max);

        for (int dim = lane; dim < head_dim; dim += kWarpSize) {
            const int v_idx =
                ((physical_block * block_size + block_offset) * kv_heads + kv_head) *
                    head_dim +
                dim;
            const int local_idx = warp_id * head_dim + dim;
            warp_out[local_idx] =
                alpha * warp_out[local_idx] + beta * __half2float(v_cache[v_idx]);
        }
        local_sum = alpha * local_sum + beta;
        local_max = new_max;
    }

    if (lane == 0) {
        warp_sum_buf[warp_id] = local_sum;
        warp_max_buf[warp_id] = local_max;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float global_max = -INFINITY;
        for (int warp = 0; warp < warps; ++warp) {
            global_max = fmaxf(global_max, warp_max_buf[warp]);
        }
        float global_sum = 0.0f;
        for (int warp = 0; warp < warps; ++warp) {
            if (warp_max_buf[warp] != -INFINITY) {
                global_sum += expf(warp_max_buf[warp] - global_max) * warp_sum_buf[warp];
            }
        }
        global_stats[0] = global_sum;
        global_stats[1] = global_max;
    }
    __syncthreads();

    const float global_sum = global_stats[0];
    const float global_max = global_stats[1];
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (int warp = 0; warp < warps; ++warp) {
            if (warp_max_buf[warp] != -INFINITY) {
                value += expf(warp_max_buf[warp] - global_max) * warp_out[warp * head_dim + dim];
            }
        }
        const int out_idx = (seq_idx * query_heads + q_head) * head_dim + dim;
        output[out_idx] = __float2half(value / global_sum);
    }
}

void reference_decode(
    const Shape& shape,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<SequenceBlocks>& sequences,
    std::vector<float>& output) {

    const int q_per_kv = shape.query_heads / shape.kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    std::fill(output.begin(), output.end(), 0.0f);

    for (int batch = 0; batch < shape.batch_size; ++batch) {
        for (int q_head = 0; q_head < shape.query_heads; ++q_head) {
            const int kv_head = q_head / q_per_kv;
            std::vector<float> scores(sequences[batch].length);
            float row_max = -std::numeric_limits<float>::infinity();

            for (int token = 0; token < sequences[batch].length; ++token) {
                float score = 0.0f;
                for (int dim = 0; dim < shape.head_dim; ++dim) {
                    const int q_idx = (batch * shape.query_heads + q_head) * shape.head_dim + dim;
                    const int k_idx =
                        ((batch * shape.max_seq_len + token) * shape.kv_heads + kv_head) *
                            shape.head_dim +
                        dim;
                    score += query[q_idx] * key[k_idx];
                }
                score *= scale;
                scores[token] = score;
                row_max = std::max(row_max, score);
            }

            float denom = 0.0f;
            for (int token = 0; token < sequences[batch].length; ++token) {
                scores[token] = std::exp(scores[token] - row_max);
                denom += scores[token];
            }

            for (int dim = 0; dim < shape.head_dim; ++dim) {
                float acc = 0.0f;
                for (int token = 0; token < sequences[batch].length; ++token) {
                    const int v_idx =
                        ((batch * shape.max_seq_len + token) * shape.kv_heads + kv_head) *
                            shape.head_dim +
                        dim;
                    acc += (scores[token] / denom) * value[v_idx];
                }
                const int out_idx = (batch * shape.query_heads + q_head) * shape.head_dim + dim;
                output[out_idx] = acc;
            }
        }
    }
}

void write_profile_json(
    const std::string& path,
    const Shape& shape,
    int total_blocks,
    int used_blocks,
    float append_ms,
    float decode_ms,
    float max_abs_error,
    float max_rel_error) {

    std::ofstream out(path);
    out << "{\n";
    out << "  \"kernel\": \"paged_kv_cache_decode\",\n";
    out << "  \"device\": \"RTX_4060_target_sm89\",\n";
    out << "  \"batch_size\": " << shape.batch_size << ",\n";
    out << "  \"query_heads\": " << shape.query_heads << ",\n";
    out << "  \"kv_heads\": " << shape.kv_heads << ",\n";
    out << "  \"head_dim\": " << shape.head_dim << ",\n";
    out << "  \"block_size\": " << shape.block_size << ",\n";
    out << "  \"max_seq_len\": " << shape.max_seq_len << ",\n";
    out << "  \"total_blocks\": " << total_blocks << ",\n";
    out << "  \"used_blocks\": " << used_blocks << ",\n";
    out << "  \"append_ms\": " << append_ms << ",\n";
    out << "  \"decode_ms\": " << decode_ms << ",\n";
    out << "  \"max_abs_error\": " << max_abs_error << ",\n";
    out << "  \"max_rel_error\": " << max_rel_error << "\n";
    out << "}\n";
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    Shape shape;
    if (argc > 1) {
        shape.max_seq_len = std::atoi(argv[1]);
    }
    if (argc > 2) {
        shape.batch_size = std::atoi(argv[2]);
    }

    if (shape.query_heads % shape.kv_heads != 0) {
        fprintf(stderr, "query_heads must be divisible by kv_heads\n");
        return EXIT_FAILURE;
    }
    if (shape.max_seq_len > shape.max_blocks_per_seq * shape.block_size) {
        fprintf(stderr, "max_seq_len exceeds max_blocks_per_seq * block_size\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    const int total_blocks = shape.batch_size * ceil_div(shape.max_seq_len, shape.block_size);
    BlockAllocator allocator(total_blocks);
    std::vector<SequenceBlocks> sequences(shape.batch_size);
    std::vector<int> h_block_tables(shape.batch_size * shape.max_blocks_per_seq, -1);
    std::vector<int> h_seq_lens(shape.batch_size, shape.max_seq_len);

    for (int batch = 0; batch < shape.batch_size; ++batch) {
        sequences[batch].length = shape.max_seq_len;
        const int blocks = ceil_div(shape.max_seq_len, shape.block_size);
        sequences[batch].blocks.reserve(blocks);
        for (int logical = 0; logical < blocks; ++logical) {
            const int physical = allocator.allocate();
            sequences[batch].blocks.push_back(physical);
            h_block_tables[batch * shape.max_blocks_per_seq + logical] = physical;
        }
    }
    const int used_blocks = total_blocks - allocator.available();

    std::mt19937 rng(1234);
    const size_t dense_kv_elems =
        static_cast<size_t>(shape.batch_size) * shape.max_seq_len * shape.kv_heads *
        shape.head_dim;
    const size_t query_elems =
        static_cast<size_t>(shape.batch_size) * shape.query_heads * shape.head_dim;

    std::vector<float> h_query_f32(query_elems);
    std::vector<float> h_key_dense_f32(dense_kv_elems);
    std::vector<float> h_value_dense_f32(dense_kv_elems);
    std::vector<half> h_query(query_elems);
    std::vector<half> h_key_dense(dense_kv_elems);
    std::vector<half> h_value_dense(dense_kv_elems);

    for (size_t i = 0; i < query_elems; ++i) {
        h_query_f32[i] = rand_uniform(rng);
        h_query[i] = __float2half(h_query_f32[i]);
    }
    for (size_t i = 0; i < dense_kv_elems; ++i) {
        h_key_dense_f32[i] = rand_uniform(rng);
        h_value_dense_f32[i] = rand_uniform(rng);
        h_key_dense[i] = __float2half(h_key_dense_f32[i]);
        h_value_dense[i] = __float2half(h_value_dense_f32[i]);
    }

    std::vector<int> h_token_offsets(shape.max_seq_len);
    std::iota(h_token_offsets.begin(), h_token_offsets.end(), 0);

    half *d_query = nullptr, *d_key_dense = nullptr, *d_value_dense = nullptr;
    half *d_k_cache = nullptr, *d_v_cache = nullptr, *d_output = nullptr;
    int *d_block_tables = nullptr, *d_seq_lens = nullptr, *d_token_offsets = nullptr;

    const size_t cache_elems =
        static_cast<size_t>(total_blocks) * shape.block_size * shape.kv_heads * shape.head_dim;

    CUDA_CHECK(cudaMalloc(&d_query, query_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_key_dense, shape.max_seq_len * shape.kv_heads * shape.head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_value_dense, shape.max_seq_len * shape.kv_heads * shape.head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, query_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, h_block_tables.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, h_seq_lens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_token_offsets, h_token_offsets.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), query_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, h_seq_lens.data(), h_seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_token_offsets, h_token_offsets.data(), h_token_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, cache_elems * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, cache_elems * sizeof(half)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float append_ms = 0.0f;
    const int kv_elems_per_seq = shape.max_seq_len * shape.kv_heads * shape.head_dim;
    const int append_threads = 256;
    const int append_blocks = ceil_div(kv_elems_per_seq, append_threads);

    CUDA_CHECK(cudaEventRecord(start));
    for (int batch = 0; batch < shape.batch_size; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * kv_elems_per_seq;
        CUDA_CHECK(cudaMemcpy(
            d_key_dense,
            h_key_dense.data() + offset,
            kv_elems_per_seq * sizeof(half),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_value_dense,
            h_value_dense.data() + offset,
            kv_elems_per_seq * sizeof(half),
            cudaMemcpyHostToDevice));
        append_kv_cache_kernel<<<append_blocks, append_threads>>>(
            d_key_dense,
            d_value_dense,
            d_token_offsets,
            d_block_tables,
            batch,
            shape.max_blocks_per_seq,
            shape.block_size,
            shape.kv_heads,
            shape.head_dim,
            shape.max_seq_len,
            d_k_cache,
            d_v_cache);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    append_ms = elapsed_ms(start, stop);

    const int warmup = 5;
    const int runs = 30;
    const dim3 decode_grid(shape.batch_size, shape.query_heads, 1);
    const dim3 decode_block(kThreads, 1, 1);
    const size_t shared_bytes =
        (shape.head_dim + (kThreads / kWarpSize) * shape.head_dim +
         2 * (kThreads / kWarpSize) + 2) *
        sizeof(float);

    for (int i = 0; i < warmup; ++i) {
        paged_decode_attention_kernel<<<decode_grid, decode_block, shared_bytes>>>(
            d_query,
            d_k_cache,
            d_v_cache,
            d_block_tables,
            d_seq_lens,
            shape.batch_size,
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.block_size,
            shape.max_blocks_per_seq,
            d_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        paged_decode_attention_kernel<<<decode_grid, decode_block, shared_bytes>>>(
            d_query,
            d_k_cache,
            d_v_cache,
            d_block_tables,
            d_seq_lens,
            shape.batch_size,
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.block_size,
            shape.max_blocks_per_seq,
            d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    const float decode_ms = elapsed_ms(start, stop) / runs;

    std::vector<half> h_output(query_elems);
    std::vector<float> h_output_f32(query_elems);
    std::vector<float> h_reference(query_elems);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, query_elems * sizeof(half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < query_elems; ++i) {
        h_output_f32[i] = __half2float(h_output[i]);
    }

    reference_decode(shape, h_query_f32, h_key_dense_f32, h_value_dense_f32, sequences, h_reference);

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < query_elems; ++i) {
        const float abs_error = std::fabs(h_output_f32[i] - h_reference[i]);
        const float rel_error = abs_error / (std::fabs(h_reference[i]) + 1e-6f);
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key_dense));
    CUDA_CHECK(cudaFree(d_value_dense));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_block_tables));
    CUDA_CHECK(cudaFree(d_seq_lens));
    CUDA_CHECK(cudaFree(d_token_offsets));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const std::string profile_path = "bench/logs/paged_kv_cache_profile.json";
    write_profile_json(
        profile_path,
        shape,
        total_blocks,
        used_blocks,
        append_ms,
        decode_ms,
        max_abs_error,
        max_rel_error);

    printf("paged KV cache validation\n");
    printf("B=%d QH=%d KVH=%d S=%d D=%d block=%d used_blocks=%d\n",
           shape.batch_size,
           shape.query_heads,
           shape.kv_heads,
           shape.max_seq_len,
           shape.head_dim,
           shape.block_size,
           used_blocks);
    printf("append_ms=%.4f decode_ms=%.4f max_abs_error=%.6f max_rel_error=%.6f\n",
           append_ms,
           decode_ms,
           max_abs_error,
           max_rel_error);
    printf("profile saved to %s\n", profile_path.c_str());

    return max_abs_error < 5e-3f ? EXIT_SUCCESS : EXIT_FAILURE;
}
