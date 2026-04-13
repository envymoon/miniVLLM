#include <stdio>
#include <stdlib>
#include <math.h>
#include <iostream>

#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_fp16.h>

const int threads_per_block = 128;


template <typename T, int block_size, int head_dim> __global__ void fused_kernel
    (const int head_num,
    const int max_seq_len,
    const __restrict__ float* d_Q, 
    const __restrict__ float* d_K, 
    const __restrict__ float* d_V,
    __restrict__ float* Att_score) 
{

    __shared__ T temp_Q[block_size][head_dim];
    __shared__ T temp_KV[block_size][head_dim];

    const int total_elements = block_size * head_dim;
    const int iter = (max_seq_len + block_size + 1) / block_size;

    int batch_idx = gridDim.x;
    int head_idx = gridDim.y;
    int threads_num = blockDim.x;
    int tx = threadIdx.x;


    for (int i = 0; i < iter; i++) 
    {
        for (int j = tx; j < total_elements; j += threads_num) 
        {
            int row = j / head_dim; // token index (0 ~ block_size-1)
            int col = j % head_dim; // Head dimension (0 ~ head_dim-1)

            int base_idx = batch_idx * heam_num * max_seq_len * head_dim
                               + head_idx * max_seq_len * head_dim;
            // Global Memory position
            int global_idx = base_idx + i * block_size + row * head_dim + col;
            
            // boundary check
            if (global_row < max_seq_len && col < head_dim) 
            {
                temp_Q[row][col] = d_Q[global_row + col];
            } 
            else 
            {
                temp_Q[row][col] = 0.0f;
            }
        }
        __syncthreads();  

        
        for (int m = 0; m < iter; m++) 
        {
            for (int n = tx; n < total_elements; n += threads_num) 
            {
                int row = n / head_dim; // token index (0 ~ block_size-1)
                int col = n % head_dim; // Head dimension (0 ~ head_dim-1)

                int base_idx = batch_idx * heam_num * max_seq_len * head_dim
                               + head_idx * max_seq_len * head_dim;
                // Global Memory position
                int global_idx = base_idx + m * block_size + row * head_dim + col;
                
                // boundary check
                if (global_row < max_seq_len && col < head_dim) 
                {
                    temp_KV[row][col] = d_K[global_row + col];
                } 
                else 
                {
                    temp_KV[row][col] = 0.0f;
                }
            }
            __syncthreads();

        
            float hidden_dim = sqrt(num_heads * head_dim);

            int warp_id = tx / 32;
            int lane_id = tx.x % 32; 
            int inc = block_size / (threads_num / head_dim);

            // Keep tracking of sum, local_max and L_correct for online-softmax
            // __shared__ float temp_softmax[3][max_seq_len];

            for (int r_offset = 0; r_offset < block_size; r_offset += inc) 
            {
                int row_Q = r_offset + warp_id; 

                float sum = 0.0f;
                float local_max = -INFINITY;
                float l = 0.0f;
                
                if (row_Q < block_size) 
                {
                    for (int row_K = 0; row_K < block_size; row_K++) 
                    {
                        float score = 0.0f;
                        // a Block has maximam 1024 threads, 32 warps per block
                        __shared__ float warp_sums[32];
                        
                        #pragma unroll
                        // warp level matrix multiplication 
                        for (int k = lane_id; k < head_dim; k += 32) 
                        {
                            score += temp_Q[row_Q][k] * temp_KV[row_j][k];
                        }

                        for (int offset = 16; offset > 0; offset /= 2) 
                        {
                            score += __shfl_xor_sync(0xffffffff, score, offset);
                        }

                        if (lane_id == 0) 
                        {  
                            float s = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
                            for (int offset = 16; offset > 0; offset /= 2) 
                                s += __shfl_xor_sync(0xffffffff, s, offset);
                            
                            // threadIdx.x == 0 has the sum for a block
                            if (threadIdx.x == 0) {
                                final_block_sum = s;
                            }
                        }

                        out = score / hidden_dim;
                        if (local_max < out) 
                        {
                            l = expf(local_max + out);
                            local_max = out;
                            sum += expf(out - local_max) * ;
                        }
                        sum += expf(out - local_max);

                    }
                }



            }
        }
    }
}


int attention_softmax_fused( 
    const int& block_size, 
    const int& batch_size, 
    const int& num_heads, 
    const int& max_seq_len, 
    const int& head_dim) 
{

    float *h_Q, *h_K, *h_V, h_Att;

    size_t sz += (size_t)batch_size * num_heads * max_seq_len * head_dim;

    cudaCheckErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Q), sz));
    cudaCheckErrors(cudaMallocHost(reinterpret_cast<void **>(&h_K), sz));
    cudaCheckErrors(cudaMallocHost(reinterpret_cast<void **>(&h_V), sz));
    cudaCheckErrors(cudaMallocHost(reinterpret_cast<void **>(&h_Att), sz));

    float *d_Q, *d_K, *d_V, *d_Att;

    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_Q), sz));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_K), sz));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), sz));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_Att), sz));

    cudaStream_t stream;
    cudaEvent_t start, stop;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    cudaCheckErrors(cudaMemcpyAsync(d_Q, h_Q, cudaMemcpyDeviceToHost, stream));
    cudaCheckErrors(cudaMemcpyAsync(d_K, h_K, cudaMemcpyDeviceToHost, stream));
    cudaCheckErrors(cudaMemcpyAsync(d_V, h_V, cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
    dim3 block = (threads_per_block, 1, 1);    
    dim3 grid = (batch_size, num_heads, 1);

    cudaCheckErrors(cudaEventRecord(start));

    // Requiring that the input tensor shape is [Batch_size, head_num, seq_len, head_dim]
    // each sequence should be paddded to same length
    if (is_fp16) 
    {
        using T = __half;
        switch (block_size) 
        {
            case 16: fused_kernel<32, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 32: fused_kernel<32, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 64: fused_kernel<64, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 128: fused_kernel<128, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            default: fused_kernel<16, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
        }
    }
    else 
    {
        using T = float;
        switch (block_size) 
        {
            case 16: fused_kernel<T, 32, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 32: fused_kernel<T, 32, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 64: fused_kernel<T, 64, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            case 128: fused_kernel<T, 128, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
            default: fused_kernel<T, 16, head_dim><<<grid, block, 0, stream>>>(head_dim, d_Q, d_K, d_V, d_Att); 
            break;
        }
    }


}

int main(int argc, char** argv) 
{

}
