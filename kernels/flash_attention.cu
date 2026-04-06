#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>  


template <int block_size> __global__ void fused_kernel(
    const __restrict__ float* d_Q, 
    const __restrict__ float* d_K, 
    const __restrict__ float* d_V,
    const float* d_Att;
) {
    
    
}


int attention_softmax_fused( 
    int block_size, 
    int batch_size, 
    int num_heads, 
    int seq_len, 
    int head_dim 
) {

    float *h_Q, *h_K, *h_V, h_Att;
    
    size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim;

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

    switch (block_size) {
        case 32: fused_kernel<32><<<>>>(d_Q, d_K, d_V, d_Att); break;
        case 16: fused_kernel<16><<<>>>(d_Q, d_K, d_V, d_Att); break;
        default: fused_kernel<16><<<>>>(d_Q, d_K, d_V, d_Att); break;
    }




}

int main(int argc, char** argv) {

}
