// ============================================================================
// 1. File Notes
// ============================================================================
// Optimized flash attention variant for Ada-class consumer GPUs.
// Design goals:
// 1. expose more parallelism by mapping the Q tile to grid.z
// 2. cut per-block shared memory footprint versus the teaching kernel
// 3. use half2 vectorization for D=64 to reduce instruction count / dependency depth
// 4. keep the implementation runnable on RTX 4060 class hardware without Tensor Core MMA

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kHeadDim = 64;
constexpr int kBlockM = 16;
constexpr int kBlockN = 16;
constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void fused_kernel_optimized_64(
    const int head_num,
    const int max_seq_len,
    const half* __restrict__ d_Q,
    const half* __restrict__ d_K,
    const half* __restrict__ d_V,
    float* __restrict__ d_Out) {

    __shared__ half2 s_q[kBlockM][kHeadDim / 2];
    __shared__ half2 s_k[kBlockN][kHeadDim / 2];
    __shared__ half2 s_v[kBlockN][kHeadDim / 2];

    const int tx = threadIdx.x;
    const int warp_id = tx / kWarpSize;
    const int lane_id = tx % kWarpSize;
    const int q_tile = blockIdx.z;
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_row_start = q_tile * kBlockM;
    const int bh_offset =
        ((batch_idx * head_num) + head_idx) * max_seq_len * kHeadDim;
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    const int kv_tiles = (max_seq_len + kBlockN - 1) / kBlockN;

    for (int idx = tx; idx < kBlockM * (kHeadDim / 2); idx += blockDim.x) {
        const int row = idx / (kHeadDim / 2);
        const int col2 = idx % (kHeadDim / 2);
        const int q_row = q_row_start + row;
        s_q[row][col2] =
            (q_row < max_seq_len)
                ? reinterpret_cast<const half2*>(d_Q + bh_offset + q_row * kHeadDim)[col2]
                : __floats2half2_rn(0.0f, 0.0f);
    }
    __syncthreads();

    for (int row_q = warp_id; row_q < kBlockM; row_q += (kThreadsPerBlock / kWarpSize)) {
        const int q_idx = q_row_start + row_q;
        if (q_idx >= max_seq_len) {
            continue;
        }

        const half2 q_vec = s_q[row_q][lane_id];
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        float2 row_out = make_float2(0.0f, 0.0f);

        for (int kv_tile = 0; kv_tile < kv_tiles; ++kv_tile) {
            const int kv_row_start = kv_tile * kBlockN;

            for (int idx = tx; idx < kBlockN * (kHeadDim / 2); idx += blockDim.x) {
                const int row = idx / (kHeadDim / 2);
                const int col2 = idx % (kHeadDim / 2);
                const int kv_row = kv_row_start + row;
                if (kv_row < max_seq_len) {
                    const half2* k_row_ptr =
                        reinterpret_cast<const half2*>(d_K + bh_offset + kv_row * kHeadDim);
                    const half2* v_row_ptr =
                        reinterpret_cast<const half2*>(d_V + bh_offset + kv_row * kHeadDim);
                    s_k[row][col2] = k_row_ptr[col2];
                    s_v[row][col2] = v_row_ptr[col2];
                } else {
                    s_k[row][col2] = __floats2half2_rn(0.0f, 0.0f);
                    s_v[row][col2] = __floats2half2_rn(0.0f, 0.0f);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int row_k = 0; row_k < kBlockN; ++row_k) {
                const int k_idx = kv_row_start + row_k;
                if (k_idx >= max_seq_len) {
                    break;
                }

                const half2 k_vec = s_k[row_k][lane_id];
                const float2 qf = __half22float2(q_vec);
                const float2 kf = __half22float2(k_vec);
                float score = qf.x * kf.x + qf.y * kf.y;
                score = warp_sum(score);
                score *= scale;
                const float score_bcast = __shfl_sync(0xffffffff, score, 0);

                const float new_max = fmaxf(row_max, score_bcast);
                const float alpha =
                    (row_max == -INFINITY) ? 0.0f : expf(row_max - new_max);
                const float beta = expf(score_bcast - new_max);

                const half2 v_vec = s_v[row_k][lane_id];
                const float2 vf = __half22float2(v_vec);
                row_out.x = alpha * row_out.x + beta * vf.x;
                row_out.y = alpha * row_out.y + beta * vf.y;
                row_sum = alpha * row_sum + beta;
                row_max = new_max;
            }
            __syncthreads();
        }

        const int out_base =
            ((batch_idx * head_num + head_idx) * max_seq_len + q_idx) * kHeadDim;
        d_Out[out_base + lane_id * 2 + 0] = row_out.x / row_sum;
        d_Out[out_base + lane_id * 2 + 1] = row_out.y / row_sum;
    }
}

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
                        output += (scores[k_row] / row_sum) *
                                  h_V[bh_offset + k_row * head_dim + d];
                    }
                    h_Out[bh_offset + q * head_dim + d] = output;
                }
            }
        }
    }
}

float run_flash_attention_kernel_optimized(
    const int batch_size,
    const int num_heads,
    const int max_seq_len,
    const int head_dim,
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Out) {

    if (head_dim != kHeadDim) {
        printf("optimized kernel only supports head_dim=%d\n", kHeadDim);
        return -1.0f;
    }

    const size_t tensor_elems =
        static_cast<size_t>(batch_size) * num_heads * max_seq_len * head_dim;
    const size_t tensor_bytes_f32 = tensor_elems * sizeof(float);
    const size_t tensor_bytes_f16 = tensor_elems * sizeof(half);

    std::vector<half> h_Q_half(tensor_elems);
    std::vector<half> h_K_half(tensor_elems);
    std::vector<half> h_V_half(tensor_elems);
    for (size_t i = 0; i < tensor_elems; ++i) {
        h_Q_half[i] = __float2half(h_Q[i]);
        h_K_half[i] = __float2half(h_K[i]);
        h_V_half[i] = __float2half(h_V[i]);
    }

    half *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float* d_Out = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_Q), tensor_bytes_f16));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_K), tensor_bytes_f16));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_V), tensor_bytes_f16));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_Out), tensor_bytes_f32));

    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaMemcpyAsync(
        d_Q, h_Q_half.data(), tensor_bytes_f16, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(
        d_K, h_K_half.data(), tensor_bytes_f16, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(
        d_V, h_V_half.data(), tensor_bytes_f16, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    const dim3 block(kThreadsPerBlock, 1, 1);
    const dim3 grid(
        batch_size,
        num_heads,
        (max_seq_len + kBlockM - 1) / kBlockM);

    checkCudaErrors(cudaEventRecord(start, stream));
    fused_kernel_optimized_64<<<grid, block, 0, stream>>>(
        num_heads, max_seq_len, d_Q, d_K, d_V, d_Out);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaMemcpyAsync(
        h_Out, d_Out, tensor_bytes_f32, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    float elapsed_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsed_ms, start, stop));

    checkCudaErrors(cudaFree(d_Q));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_Out));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    return elapsed_ms;
}

int validate_and_benchmark_flash_attention_optimized() {
    const int batch_size = 4;
    const int num_heads = 8;
    const int max_seq_len = 128;
    const int head_dim = kHeadDim;
    const int warmup = 5;
    const int runs = 20;

    const size_t tensor_elems =
        static_cast<size_t>(batch_size) * num_heads * max_seq_len * head_dim;
    const size_t tensor_bytes = tensor_elems * sizeof(float);

    float *h_Q = nullptr, *h_K = nullptr, *h_V = nullptr;
    float *h_Out = nullptr, *h_Ref = nullptr;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_Q), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_K), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_V), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_Out), tensor_bytes));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_Ref), tensor_bytes));

    for (size_t i = 0; i < tensor_elems; ++i) {
        h_Q[i] = sinf(0.013f * static_cast<float>(i));
        h_K[i] = cosf(0.017f * static_cast<float>(i));
        h_V[i] = sinf(0.019f * static_cast<float>(i + 11));
        h_Out[i] = 0.0f;
        h_Ref[i] = 0.0f;
    }

    reference_attention(
        batch_size, num_heads, max_seq_len, head_dim, h_Q, h_K, h_V, h_Ref);

    for (int i = 0; i < warmup; ++i) {
        run_flash_attention_kernel_optimized(
            batch_size, num_heads, max_seq_len, head_dim, h_Q, h_K, h_V, h_Out);
    }

    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        total_ms += run_flash_attention_kernel_optimized(
            batch_size, num_heads, max_seq_len, head_dim, h_Q, h_K, h_V, h_Out);
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

    printf("flash attention optimized validation\n");
    printf(
        "B=%d H=%d S=%d D=%d block_m=%d block_n=%d threads=%d\n",
        batch_size,
        num_heads,
        max_seq_len,
        head_dim,
        kBlockM,
        kBlockN,
        kThreadsPerBlock);
    printf("max_abs_error=%.8f max_rel_error=%.8f has_nan=%s\n",
           max_abs_error,
           max_rel_error,
           has_nan ? "true" : "false");
    printf("avg_time_ms=%.4f estimated_TFLOPs=%.4f\n", avg_ms, tflops);

    checkCudaErrors(cudaFreeHost(h_Q));
    checkCudaErrors(cudaFreeHost(h_K));
    checkCudaErrors(cudaFreeHost(h_V));
    checkCudaErrors(cudaFreeHost(h_Out));
    checkCudaErrors(cudaFreeHost(h_Ref));

    return has_nan ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage: flash_attention_optimized sample\n");
        return EXIT_SUCCESS;
    }

    findCudaDevice(argc, (const char**)argv);
    return validate_and_benchmark_flash_attention_optimized();
}
