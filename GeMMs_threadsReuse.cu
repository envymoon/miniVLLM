/* 
TaskLists:
* Finish up GeMMs 
* Finish up and understand main()
* Insert matmul_naive
* Debug
* Run it with Profiler to learn Hardward Stats
*/


#include <cuda_runtime.h> 
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>    

__global__ void matMul(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


template <int tile_size> __global__ void tiling_matMul (
    const int n, 
    const int m, 
    const int k, 
    float * __restrict__ d_c, 
    const float * __restrict__ d_a, 
    const float * __restrict__ d_b) {
    
    __shared__ float tile_A[tile_size][tile_size];
    __shared__ float tile_B[tile_size][tile_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // left and up corner, row and col positions of a thread in Matrix C from current block                                                         
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;    
    
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    for (int t = 0; t < (k + tile_size - 1) / tile_size; t++) {
        // left and up corner, row and col positions of a thread
        int a_row = (blockIdx.y * 16 + ty) * 2;
        int a_col = t * 32 + tx * 2;

        tile_A[ty * 2][tx * 2]         = (a_row < m && a_col < k) ? d_a[a_row * k + a_col] : 0.0f;
        tile_A[ty * 2][tx * 2 + 1]     = (a_row < m && a_col + 1 < k) ? d_a[a_row * k + a_col + 1] : 0.0f;
        tile_A[ty * 2 + 1][tx * 2]     = (a_row + 1 < m && a_col < k) ? d_a[(a_row + 1) * k + a_col] : 0.0f;
        tile_A[ty * 2 + 1][tx * 2 + 1] = (a_row + 1 < m && a_col + 1 < k) ? d_a[(a_row + 1) * k + a_col + 1] : 0.0f;
        
        // left and up corner, row and col positions of a thread
        int b_row = t * 32 + ty * 2;
        int b_col = (blockIdx.x * 16 + tx) * 2;

        tile_B[ty * 2][tx * 2]         = (b_row < k && b_col < n) ? d_b[b_base_row * n + b_base_col] : 0.0f;
        tile_B[ty * 2][tx * 2 + 1]     = (b_row < k && b_col + 1 < n) ? d_b[b_base_row * n + b_base_col + 1] : 0.0f;
        tile_B[ty * 2 + 1][tx * 2]     = (b_row + 1 < k && b_col < n) ? d_b[(b_base_row + 1) * n + b_base_col] : 0.0f;
        tile_B[ty * 2 + 1][tx * 2 + 1] = (b_row + 1 < k && b_col + 1 < n) ? d_b[(b_base_row + 1) * n + b_base_col + 1] : 0.0f;

        __syncthreads();

  
#pragma unroll
        for (int j = 0; j < tile_size; j++) {
            float a0 = tile_A[threadIdx.y * 2][j];
            float a1 = tile_A[threadIdx.y * 2 + 1][j];
            float b0 = tile_B[j][threadIdx.x * 2];
            float b1 = tile_B[j][threadIdx.x * 2 + 1];
            
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        // sync for outer for loop
        __syncthreads();
    }

    if (row < m && col < n) d_c[row * n + col] = sum00;
    if (row < m && col + 1 < n) d_c[row * n + col + 1] = sum01;
    if (row + 1 < m && col < n) d_c[(row + 1) * n + col] = sum10;
    if (row + 1 < m && col + 1 < n) d_c[(row + 1) * n + col + 1] = sum11;
}



int GeMMs(int argc, char **argv, int tile_size, const dim3 &dim_a, const dim3 &dim_b) {
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

    int t_size = tile_size / 2
    dim3 block = (t_size, t_size, 1);
    dim3 grid  = (dim_c.x/t_size, dim_c.y/t_size, 1);

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

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecTotal,
           flopsPerMatrixMul,
           threads.x * threads.y);


    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
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
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

void matMul_naive() {

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