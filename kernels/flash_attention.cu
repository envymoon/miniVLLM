#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>  
#include <cuda_fp16.h>


template <typename T>
__device__ __forceinline__ T cast_to(float val);

// float type conversion
template <>
__device__ __forceinline__ float cast_to<float>(float val) { return val; }

// half type conversion
template <>
__device__ __forceinline__ __half cast_to<__half>(float val) { return __float2half(val); }
 

template <typename T, int block_size, int head_dim> __global__ void fused_kernel(
    // const int num_heads, 
    const int max_seq_len, 
    const __restrict__ float* d_Q, 
    const __restrict__ float* d_K, 
    const __restrict__ float* d_V,
    __restrict__ float* d_Att
) {
    __shared__ T t_Q[block_size][head_dim];
    __shared__ T t_KV[block_size][head_dim];
    //__shared__ float t_att[block_size][head_dim];

    int batch_idx = gridDim.x;
    int head_idx = gridDim.y;
    int block_idx = gridDim.z;

    float local_m = -INFINITY;
    float local_l = 0.0f;

    for (int i = 0; i < max_seq_len; i++) {
        int q_x = blockIdx.x * 4;
        int tx = threadIdx.x;
        t_Q[q_x][tx] =  
        t_Q[q_x + 1][tx] = 
        t_Q[q_x + 2][tx] = 
        t_Q[q_x + 3][tx] = 


    }
     
}


int attention_softmax_fused( 
    const int& block_size, 
    const int& batch_size, 
    const int& num_heads, 
    const int& max_seq_len, 
    const int& head_dim
) {

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

    int block_serializing = block_size / 4
    int num_tiles = (max_seq_len + block_size - 1) / (block_serializing);
    dim3 block = (block_serializing, 1, 1);    
    dim3 grid = (batch_size, num_heads, num_tiles);

    cudaCheckErrors(cudaEventRecord);

    // Requiring that the input tensor dimension is [Batch_size, head_num, seq_len, head_dim]
    // each sequence should be paddded to same length
    if (is_fp16) {
        using T = __half;
        switch (block_size) {
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
    else {
        using T = float;
        switch (block_size) {
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

int main(int argc, char** argv) {

}
