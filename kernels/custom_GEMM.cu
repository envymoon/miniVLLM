#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>      


template <int tile_size> __global__ void tiling_Matrix (const int n, 
                                                        const int m, 
                                                        const int k, 
                                                        float * __restrict__ d_c, 
                                                        const float * __restrict__ d_a, 
                                                        const float * __restrict__ d_b) {
    
    __shared__ float tile_A[tile_size][tile_size];
    __shared__ float tile_B[tile_size][tile_size];


    // row and col positions of threads in Matrix C from current block                                                         
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;    
    
 
    float sum = 0.0f;

    for (int t = 0; t < (k + tile_size - 1) / tile_size; t++) {
        int a_col = t * tile_size + threadIdx.x;
        int b_row = t * tile_size + threadIdx.y;

        if (row < m && a_col < k)
            tile_A[threadIdx.y][threadIdx.x] = d_a[row * k + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && b_row < k)
            tile_B[threadIdx.y][threadIdx.x] = d_b[b_row * n + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

  
        #pragma unroll
        for (int j = 0; j < tile_size; j++) {
            sum += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        // sync for outer for loop
        __syncthreads();
    }

    if (row < m && col < n) {
        d_c[row * n + col] = sum;
    }


}



void GeMMs(int argc, char **argv, int tile_size, const dim3 &dim_a, const dim3 &dim_b) {
    dim3   dim_c = (dim_b.x, dim_a.y, 1);

    float* h_c, *h_b, *h_a;
    float* d_c, *d_b, *d_a;

    size_t sz_a = dim_a.x * dim_a.y * sizeof(float);
    size_t sz_b = dim_b.x * dim_b.y * sizeof(float);
    size_t sz_c = dim_c.x * dim_c.y * sizeof(float);

    checkCudaErrors(cudaMallocHost(&h_a, sz_a));
    checkCudaErrors(cudaMallocHost(&h_b, sz_b));
    checkCudaErrors(cudaMallocHost(&h_c, sz_c));

    cudaStream_t stream;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_a), sz_a));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_b), sz_b));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_c), sz_c));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // events are uses for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaMemcpyAsync(d_a, h_a, sz_a, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sz_b, cudaMemcpyHostToDevice, stream));

    dim3 block = (tile_size, tile_size, 1);
    dim3 grid  = (dim_c.x/tile_size, dim_c.y/tile_size, 1);

    checkCudaErrors(cudaStreamSynchronize(stream));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    switch (tile_size) {
        case 32: tiling_Matrix<32><<<grid, block, 0, stream>>>(dim_a.y, dim_b.x, dim_a.x, d_c, d_a, d_b); break;
        case 16: tiling_Matrix<16><<<grid, block, 0, stream>>>(dim_a.y, dim_b.x, dim_a.x, d_c, d_a, d_b); break;
        default: tiling_Matrix<16><<<grid, block, 0, stream>>>(dim_a.y, dim_b.x, dim_a.x, d_c, d_a, d_b); break;
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaMemcpyAsync(h_c, d_c, sz_c, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));


    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6; // machine zero

    for (int i = 0; i < static_cast<int>(dim_c.x * dim_c.y); i++) {
        double abs_err    = fabs(h_c[i] - (dim_a.x * valB));
        double dot_length = dim_a.x;
        double abs_val    = fabs(h_C[i]);
        double rel_err    = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}


int main(int argc, char **argv) {
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("M x K (Width x Height of Matrix A)\n");
        printf("N x K (Width x Height of Matrix B)\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int tile_size;

}