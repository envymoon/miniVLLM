#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

const int threads_per_tile = 128;

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

template <typename T, int block_size, int head_dim>
__global__ void fused_kernel(
    const int head_num,
    const int max_seq_len,
    const float* __restrict__ d_Q,
    const float* __restrict__ d_K,
    const float* __restrict__ d_V,
    float* __restrict__ Att_score) {

    __shared__ T temp_Q[block_size][head_dim];
    __shared__ T temp_K[block_size][head_dim];
    __shared__ T temp_V[block_size][head_dim];

    const int tx = threadIdx.x;
    const int warp_id = tx / warpSize;
    const int lane_id = tx % warpSize;
    const int warp_count = blockDim.x / warpSize;
    const int total_elements = block_size * head_dim;
    const int tile_count = (max_seq_len + block_size - 1) / block_size;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int bh_offset =
        ((batch_idx * head_num) + head_idx) * max_seq_len * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    constexpr int cols_per_lane = (head_dim + warpSize - 1) / warpSize;

    for (int q_tile = 0; q_tile < tile_count; ++q_tile) {
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

        for (int q_base = 0; q_base < block_size; q_base += warp_count) {
            const int row_q = q_base + warp_id;
            const int q_idx = q_tile * block_size + row_q;

            float row_max = -INFINITY;
            float row_sum = 0.0f;
            float row_out[cols_per_lane];

            #pragma unroll
            for (int i = 0; i < cols_per_lane; ++i) {
                row_out[i] = 0.0f;
            }

            for (int kv_tile = 0; kv_tile < tile_count; ++kv_tile) {
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
                    for (int row_k = 0; row_k < block_size; ++row_k) {
                        const int k_idx = kv_tile * block_size + row_k;
                        if (k_idx >= max_seq_len) {
                            continue;
                        }

                        float score = 0.0f;
                        #pragma unroll
                        for (int k = lane_id; k < head_dim; k += warpSize) {
                            score += convert_to_float(temp_Q[row_q][k]) *
                                     convert_to_float(temp_K[row_k][k]);
                        }

                        #pragma unroll
                        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                            score += __shfl_xor_sync(0xffffffff, score, offset);
                        }

                        score *= scale;
                        const float score_bcast =
                            __shfl_sync(0xffffffff, score, 0);

                        const float new_max = fmaxf(row_max, score_bcast);
                        const float alpha =
                            (row_max == -INFINITY) ? 0.0f : expf(row_max - new_max);
                        const float beta = expf(score_bcast - new_max);

                        row_sum = alpha * row_sum + beta;

                        #pragma unroll
                        for (int d = lane_id; d < head_dim; d += warpSize) {
                            const int local_col = d / warpSize;
                            row_out[local_col] =
                                alpha * row_out[local_col] +
                                beta * convert_to_float(temp_V[row_k][d]);
                        }

                        row_max = new_max;
                    }
                }
                __syncthreads();
            }

            if (row_q < block_size && q_idx < max_seq_len && row_sum > 0.0f) {
                #pragma unroll
                for (int d = lane_id; d < head_dim; d += warpSize) {
                    const int local_col = d / warpSize;
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

    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_Q), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_K), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_V), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_Att), tensor_bytes));

    for (size_t i = 0; i < tensor_elems; ++i) {
        h_Q[i] = 0.0f;
        h_K[i] = 0.0f;
        h_V[i] = 0.0f;
        h_Att[i] = 0.0f;
    }

    float *d_Q, *d_K, *d_V, *d_Att;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_Q), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_K), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_V), tensor_bytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_Att), tensor_bytes));

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
    printf("fused attention kernel finished in %.3f ms\n", elapsed_ms);

    checkCudaErrors(cudaFreeHost(h_Q));
    checkCudaErrors(cudaFreeHost(h_K));
    checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_Att));
    checkCudaErrors(cudaFree(d_Q));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_Att));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage: flash_attention sample\n");
        return EXIT_SUCCESS;
    }

    findCudaDevice(argc, (const char**)argv);
    return EXIT_SUCCESS;
}
