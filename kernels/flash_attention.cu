#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>  


template <int block_size> __global__ void flash_attention() {

}


int attention(
    int argc, 
    char **argv, 
    int block_size, 
    int batch_size, 
    int num_heads, 
    int seq_len, 
    int head_dim 
) {

    float *d_Q, *d_K, *d_V, *att_K;
    size_t h_sz = (size_t)batch_size * num_heads * seq_len * head_dim;
    cudaCheckErrors();


    std::vector<float> h_Q(h_sz);
    std::vector<float> h_K(h_sz);
    std::vector<float> h_V(h_sz);
    std::vector<float> att_V(h_sz);

    float *d_Q, *d_K, *d_V, *att_K;

    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(), ));


}
