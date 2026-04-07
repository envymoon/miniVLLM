#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>  


template <int block_size, int head_dim> __global__ void fused_kernel(
    const int num_heads, 
    const int* seq_len, 
    const __restrict__ float* d_Q, 
    const __restrict__ float* d_K, 
    const __restrict__ float* d_V,
    __restrict__ float* d_Att
) {
    __shared__ t_Q[block_size][head_dim];
    __shared__ t_K[block_size][head_dim];
    __shared__ t_V[block_size][head_dim];

    int counter = seq_lenlen / block_size;

    for (int i = 0; i < counter; i++) {
        int q_x = 4 * blockIdx.x;
        t_Q[q_x][] = 
        t_Q[q_x + 1][] = 
        t_Q[q_x + 2][] = 
        t_Q[q_x + 3][] = 

    }
     
}


int attention_softmax_fused(
    const int& total_token, 
    const int& block_size, 
    const int& batch_size, 
    const int& num_heads, 
    const int* seq_len, 
    const int& head_dim
) {

    float *h_Q, *h_K, *h_V, h_Att;
    
    size_t sz = (size_t)batch_size * num_heads * cumu_seq_len * head_dim;

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

    dim3 block = (block_size / 4, 1, 1);    
    dim3 grid = (batch_size, num_heads, 1);

    // Assuming that the input tensor dimension is [Batch_size, head_num, seq_len, head_dim]
    switch (block_size) {
        case 64: fused_kernel<32, head_dim><<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_Att); break;
        case 128: fused_kernel<128, head_dim><<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_Att); break;
        case 256: fused_kernel<256, head_dim><<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_Att); break;
        default: fused_kernel<64, head_dim><<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_Att); break;
    }




}

int main(int argc, char** argv) {

}
