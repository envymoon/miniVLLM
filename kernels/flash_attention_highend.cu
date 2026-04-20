// ============================================================================
// 1. File Notes
// ============================================================================
// Current fused attention implementation.
// Notes:
// 1. current output layout is [B, H, S, D]
// 2. this high-end variant keeps the wider dispatch range
// 3. intended for devices that can tolerate larger shared-memory footprints
// 4. KV cache / paged attention path is not implemented yet

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector>

const int threads_per_tile = 128;
constexpr int warp_size_const = 32;

// ============================================================================
// 2. Device Helpers
// ============================================================================

template <typename T>
__device__ __forceinline__ T convert_from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half convert_from_float<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__device__ __forceinline__ float convert_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float convert_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 convert_from_float<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float convert_to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

// ============================================================================
// 3. Core Fused Attention Kernel
// ============================================================================

template <typename T, int block_size, int head_dim>
__global__ void fused_kernel(
    const int head_num,
    const int max_seq_len,
    const float* __restrict__ d_Q,
    const float* __restrict__ d_K,
    const float* __restrict__ d_V,
    float* __restrict__ Att_score) {

    // Shared-memory tiles for one Q tile and one KV tile.
    __shared__ T temp_Q[block_size][head_dim];
    __shared__ T temp_K[block_size][head_dim];
    __shared__ T temp_V[block_size][head_dim];

    // Basic thread / warp decomposition inside one block.
    const int tx = threadIdx.x;
    const int warp_id = tx / warp_size_const;
    const int lane_id = tx % warp_size_const;
    const int warp_count = blockDim.x / warp_size_const;
    const int total_elements = block_size * head_dim;
    const int tile_count = (max_seq_len + block_size - 1) / block_size;

    // Each block is assigned to one (batch, head) pair.
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int bh_offset =
        ((batch_idx * head_num) + head_idx) * max_seq_len * head_dim;
    // Standard attention scaling term: 1 / sqrt(head_dim).
    const float scale = rsqrtf(static_cast<float>(head_dim));

    // Each lane keeps a small slice of the output vector in registers.
    constexpr int cols_per_lane = (head_dim + warp_size_const - 1) / warp_size_const;

    // Outer loop over Q tiles.
    for (int q_tile = 0; q_tile < tile_count; ++q_tile) {
        // Load one block_size x head_dim tile of Q into shared memory.
        for (int idx = tx; idx < total_elements; idx += blockDim.x) {
            const int row = idx / head_dim;
            const int col = idx % head_dim;
            const int q_row = q_tile * block_size + row;

            if (q_row < max_seq_len) {
                const int global_idx = bh_offset + q_row * head_dim + col;
                temp_Q[row][col] = convert_from_float<T>(d_Q[global_idx]);
            } else {
                temp_Q[row][col] = convert_from_float<T>(0.0f);
            }
        }
        __syncthreads();

        // One warp works on one query row at a time.
        for (int q_base = 0; q_base < block_size; q_base += warp_count) {
            const int row_q = q_base + warp_id;
            const int q_idx = q_tile * block_size + row_q;

            // Online softmax state for one query row.
            float row_max = -INFINITY;
            float row_sum = 0.0f;
            float row_out[cols_per_lane];

            #pragma unroll
            for (int i = 0; i < cols_per_lane; ++i) {
                row_out[i] = 0.0f;
            }

            // Sweep across all KV tiles for the current Q tile.
            for (int kv_tile = 0; kv_tile < tile_count; ++kv_tile) {
                // Load one K tile and the matching V tile into shared memory.
                for (int idx = tx; idx < total_elements; idx += blockDim.x) {
                    const int row = idx / head_dim;
                    const int col = idx % head_dim;
                    const int kv_row = kv_tile * block_size + row;

                    if (kv_row < max_seq_len) {
                        const int global_idx = bh_offset + kv_row * head_dim + col;
                        temp_K[row][col] = convert_from_float<T>(d_K[global_idx]);
                        temp_V[row][col] = convert_from_float<T>(d_V[global_idx]);
                    } else {
                        temp_K[row][col] = convert_from_float<T>(0.0f);
                        temp_V[row][col] = convert_from_float<T>(0.0f);
                    }
                }
                __syncthreads();

                if (row_q < block_size && q_idx < max_seq_len) {
                    // Iterate over all key rows inside the current KV tile.
                    for (int row_k = 0; row_k < block_size; ++row_k) {
                        const int k_idx = kv_tile * block_size + row_k;
                        if (k_idx >= max_seq_len) {
                            continue;
                        }

                        // Warp-level dot product between Q[row_q, :] and K[row_k, :].
                        float score = 0.0f;
                        #pragma unroll
                        for (int k = lane_id; k < head_dim; k += warp_size_const) {
                            score += convert_to_float(temp_Q[row_q][k]) *
                                     convert_to_float(temp_K[row_k][k]);
                        }

                        // Reduce the partial dot products across the warp.
                        #pragma unroll
                        for (int offset = warp_size_const / 2; offset > 0; offset /= 2) {
                            score += __shfl_xor_sync(0xffffffff, score, offset);
                        }

                        score *= scale;
                        // Lane 0 owns the scalar score; broadcast it to the whole warp.
                        const float score_bcast =
                            __shfl_sync(0xffffffff, score, 0);

                        // Online softmax update:
                        // keep the running max, denominator, and output numerator.
                        const float new_max = fmaxf(row_max, score_bcast);
                        const float alpha =
                            (row_max == -INFINITY) ? 0.0f : expf(row_max - new_max);
                        const float beta = expf(score_bcast - new_max);

                        row_sum = alpha * row_sum + beta;

                        // Each lane updates its own slice of the output vector.
                        #pragma unroll
                        for (int d = lane_id; d < head_dim; d += warp_size_const) {
                            const int local_col = d / warp_size_const;
                            row_out[local_col] =
                                alpha * row_out[local_col] +
                                beta * convert_to_float(temp_V[row_k][d]);
                        }

                        row_max = new_max;
                    }
                }
                __syncthreads();
            }

            // Write the normalized output back as a flattened [B, H, S, D] tensor.
            if (row_q < block_size && q_idx < max_seq_len && row_sum > 0.0f) {
                #pragma unroll
                for (int d = lane_id; d < head_dim; d += warp_size_const) {
                    const int local_col = d / warp_size_const;
                    const int out_idx =
                        ((batch_idx * head_num + head_idx) * max_seq_len + q_idx) *
                            head_dim +
                        d;
                    Att_score[out_idx] = row_out[local_col] / row_sum;
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// 4. Launch and Runtime Wrappers
// ============================================================================

template <typename T, int head_dim>
void launch_block_size_kernel(
    const int block_size,
    const dim3& grid,
    const dim3& block,
    cudaStream_t stream,
    const int head_num,
    const int max_seq_len,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_Att) {

    // Dispatch on block_size so block_size can stay a compile-time constant
    // inside the kernel for shared-memory layout and loop unrolling.
    switch (block_size) {
        case 16:
            fused_kernel<T, 16, head_dim><<<grid, block, 0, stream>>>(
                head_num, max_seq_len, d_Q, d_K, d_V, d_Att);
            break;
        case 32:
            fused_kernel<T, 32, head_dim><<<grid, block, 0, stream>>>(
                head_num, max_seq_len, d_Q, d_K, d_V, d_Att);
            break;
        case 64:
            fused_kernel<T, 64, head_dim><<<grid, block, 0, stream>>>(
                head_num, max_seq_len, d_Q, d_K, d_V, d_Att);
            break;
        case 128:
            fused_kernel<T, 128, head_dim><<<grid, block, 0, stream>>>(
                head_num, max_seq_len, d_Q, d_K, d_V, d_Att);
            break;
        default:
            printf("Unsupported block_size=%d. Use one of {16, 32, 64, 128}.\n",
                   block_size);
            break;
    }
}

template <typename T>
void launch_head_dim_kernel(
    const int block_size,
    const int head_dim,
    const dim3& grid,
    const dim3& block,
    cudaStream_t stream,
    const int head_num,
    const int max_seq_len,
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_Att) {

    // Dispatch on head_dim for the same reason:
    // compile-time head_dim enables static shared-memory shapes.
    switch (head_dim) {
        case 32:
            launch_block_size_kernel<T, 32>(
                block_size, grid, block, stream, head_num, max_seq_len, d_Q, d_K, d_V,
                d_Att);
            break;
        case 64:
            launch_block_size_kernel<T, 64>(
                block_size, grid, block, stream, head_num, max_seq_len, d_Q, d_K, d_V,
                d_Att);
            break;
        case 128:
            launch_block_size_kernel<T, 128>(
                block_size, grid, block, stream, head_num, max_seq_len, d_Q, d_K, d_V,
                d_Att);
            break;
        default:
            printf("Unsupported head_dim=%d. Use one of {32, 64, 128}.\n",
                   head_dim);
            break;
    }
}

float run_flash_attention_kernel(
    const int block_size,
    const int batch_size,
    const int num_heads,
    const int max_seq_len,
    const int head_dim,
    const bool is_fp16,
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Att) {

    const size_t tensor_elems =
        static_cast<size_t>(batch_size) * num_heads * max_seq_len * head_dim;
    const size_t tensor_bytes = tensor_elems * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_Att;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Q), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_K), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Att), tensor_bytes));

    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(
        cudaMemcpyAsync(d_Q, h_Q, tensor_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
        cudaMemcpyAsync(d_K, h_K, tensor_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
        cudaMemcpyAsync(d_V, h_V, tensor_bytes, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    const dim3 block(threads_per_tile, 1, 1);
    const dim3 grid(batch_size, num_heads, 1);

    checkCudaErrors(cudaEventRecord(start, stream));

    if (is_fp16) {
        launch_head_dim_kernel<__half>(
            block_size, head_dim, grid, block, stream, num_heads, max_seq_len, d_Q,
            d_K, d_V, d_Att);
    } else {
        launch_head_dim_kernel<float>(
            block_size, head_dim, grid, block, stream, num_heads, max_seq_len, d_Q,
            d_K, d_V, d_Att);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(
        cudaMemcpyAsync(h_Att, d_Att, tensor_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    float elapsed_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_ms, start, stop));

    checkCudaErrors(cudaFree(d_Q));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_Att));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    return elapsed_ms;
}

// ============================================================================
// 5. Reference and Benchmark
// ============================================================================

void reference_attention(
    const int batch_size,
    const int num_heads,
    const int max_seq_len,
    const int head_dim,
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Out) {

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            const int bh_offset = ((b * num_heads) + h) * max_seq_len * head_dim;

            for (int q = 0; q < max_seq_len; ++q) {
                std::vector<float> scores(max_seq_len, 0.0f);
                float row_max = -INFINITY;

                for (int k_row = 0; k_row < max_seq_len; ++k_row) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += h_Q[bh_offset + q * head_dim + d] *
                                 h_K[bh_offset + k_row * head_dim + d];
                    }
                    score *= scale;
                    scores[k_row] = score;
                    row_max = fmaxf(row_max, score);
                }

                float row_sum = 0.0f;
                for (int k_row = 0; k_row < max_seq_len; ++k_row) {
                    scores[k_row] = expf(scores[k_row] - row_max);
                    row_sum += scores[k_row];
                }

                for (int d = 0; d < head_dim; ++d) {
                    float output = 0.0f;
                    for (int k_row = 0; k_row < max_seq_len; ++k_row) {
                        const float prob = scores[k_row] / row_sum;
                        output += prob * h_V[bh_offset + k_row * head_dim + d];
                    }
                    h_Out[bh_offset + q * head_dim + d] = output;
                }
            }
        }
    }
}

int validate_and_benchmark_flash_attention() {
    const int block_size = 32;
    const int batch_size = 1;
    const int num_heads = 2;
    const int max_seq_len = 32;
    const int head_dim = 64;
    const bool is_fp16 = false;
    const int warmup = 5;
    const int runs = 20;

    const size_t tensor_elems =
        static_cast<size_t>(batch_size) * num_heads * max_seq_len * head_dim;
    const size_t tensor_bytes = tensor_elems * sizeof(float);

    float *h_Q, *h_K, *h_V, *h_Out, *h_Ref;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Q), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_K), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_V), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Out), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Ref), tensor_bytes));

    for (size_t i = 0; i < tensor_elems; ++i) {
        h_Q[i] = sinf(0.1f * static_cast<float>(i));
        h_K[i] = cosf(0.05f * static_cast<float>(i));
        h_V[i] = sinf(0.07f * static_cast<float>(i + 3));
        h_Out[i] = 0.0f;
        h_Ref[i] = 0.0f;
    }

    reference_attention(
        batch_size, num_heads, max_seq_len, head_dim, h_Q, h_K, h_V, h_Ref);

    for (int i = 0; i < warmup; ++i) {
        run_flash_attention_kernel(
            block_size, batch_size, num_heads, max_seq_len, head_dim, is_fp16, h_Q,
            h_K, h_V, h_Out);
    }

    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        total_ms += run_flash_attention_kernel(
            block_size, batch_size, num_heads, max_seq_len, head_dim, is_fp16, h_Q,
            h_K, h_V, h_Out);
    }

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    bool has_nan = false;

    for (size_t i = 0; i < tensor_elems; ++i) {
        const float abs_error = fabsf(h_Out[i] - h_Ref[i]);
        const float rel_error = abs_error / (fabsf(h_Ref[i]) + 1e-6f);
        max_abs_error = fmaxf(max_abs_error, abs_error);
        max_rel_error = fmaxf(max_rel_error, rel_error);
        if (isnan(h_Out[i]) || isinf(h_Out[i])) {
            has_nan = true;
        }
    }

    const float avg_ms = total_ms / runs;
    const double flops =
        4.0 * static_cast<double>(batch_size) * num_heads * max_seq_len *
        max_seq_len * head_dim;
    const double tflops = (flops * 1.0e-12) / (avg_ms / 1000.0);

    printf("flash attention validation\n");
    printf("B=%d H=%d S=%d D=%d block=%d\n",
           batch_size, num_heads, max_seq_len, head_dim, block_size);
    printf("max_abs_error=%.8f max_rel_error=%.8f has_nan=%s\n",
           max_abs_error, max_rel_error, has_nan ? "true" : "false");
    printf("avg_time_ms=%.4f estimated_TFLOPs=%.4f\n", avg_ms, tflops);

    checkCudaErrors(cudaFreeHost(h_Q));
    checkCudaErrors(cudaFreeHost(h_K));
    checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_Out));
    checkCudaErrors(cudaFreeHost(h_Ref));

    return EXIT_SUCCESS;
}

// ============================================================================
// 6. Public Entry and Demo
// ============================================================================

int attention_softmax_fused(
    const int& block_size,
    const int& batch_size,
    const int& num_heads,
    const int& max_seq_len,
    const int& head_dim,
    bool is_fp16) {

    float *h_Q, *h_K, *h_V, *h_Att;
    const size_t tensor_elems =
        static_cast<size_t>(batch_size) * num_heads * max_seq_len * head_dim;
    const size_t tensor_bytes = tensor_elems * sizeof(float);

    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Q), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_K), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_V), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Att), tensor_bytes));

    for (size_t i = 0; i < tensor_elems; ++i) {
        h_Q[i] = 0.0f;
        h_K[i] = 0.0f;
        h_V[i] = 0.0f;
        h_Att[i] = 0.0f;
    }

    float elapsed_ms = run_flash_attention_kernel(
        block_size, batch_size, num_heads, max_seq_len, head_dim, is_fp16, h_Q, h_K,
        h_V, h_Att);
    printf("fused attention kernel finished in %.3f ms\n", elapsed_ms);

    checkCudaErrors(cudaFreeHost(h_Q));
    checkCudaErrors(cudaFreeHost(h_K));
    checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_Att));

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage: flash_attention sample\n");
        return EXIT_SUCCESS;
    }

    findCudaDevice(argc, (const char**)argv);

    return validate_and_benchmark_flash_attention();
}
