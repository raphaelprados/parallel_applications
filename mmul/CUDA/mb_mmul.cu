
#include<iostream>
#include<cmath>
#include<cstdlib>
#include<cassert>
#include<chrono>

inline cudaError_t checkCuda(cudaError_t result) {
    if(result != cudaSuccess) {
        std::cerr << "CUDA RUntime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);          // Terminates the program if the condition is false
    }
    return result;
}

void initData(double *a, int N) {
    std::srand(42);  // Seed with 42

    for (int i = 0; i < N; i++) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;  // Generates value in [0, 1]
    }
}

__global__ void vector_mult_kernel(double *A, double *B, double *C, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x,
        grid_stride = gridDim.x * blockDim.x;
    int i, j;

    for(int l = index; l < N; l += grid_stride) {
        i = l / N;      // Row index
        j = l % N;      // Column index

        C[i*N+j] = 0;
        for(int k = 0; k < N; k++)  {
            C[i*N+j] += A[i*N+k] * B[k*N+j];
        }
    }
} 

int main(int argc, char const *argv[]) {
    
    // ------------------------------ Variable declaration ------------------------------

    // Arrays
    double *a, *b, *c;

    if(argc != 3) {
        printf("Program usage: ./bin N F, where\n"
               "   B is the square matrix dimension\n"
               "   F is the folding factor");
    }
    
    // Control variables
    int N = atoi(argv[1]),
        block_folding_factor = (int)pow(2,(double)atoi(argv[2]));

    // CUDA error variables
    cudaError_t sync_err[3], async_err;
    bool allocation_error = false;
    
    // CUDA event variables. Will be used here for time recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Execution time variables
    float kernel_time;

    // Kernel parameters
    unsigned size = N * N * sizeof(double),
             cuda_threads = 32 * block_folding_factor,
             blocks = N / cuda_threads + (N % cuda_threads != 0 ? 1 : 0);

    // ------------------------------ Memory Allocation ------------------------------

    auto begin = std::chrono::high_resolution_clock::now();

    sync_err[0] = cudaMallocManaged(&a, size);      // Checks for memory allocation errors
    sync_err[1] = cudaMallocManaged(&b, size);
    sync_err[2] = cudaMallocManaged(&c, size);

    for(int i = 0; i < 3; i++)
        allocation_error = allocation_error || (sync_err[i] != cudaSuccess);

    if(allocation_error)
        exit(-1);

    initData(a, N*N);
    initData(b, N*N);

    // ------------------------------ Processing ------------------------------

    if(!allocation_error) {
        cudaEventRecord(start);
            vector_mult_kernel<<<blocks, cuda_threads>>>(a, b, c, N);
            cudaDeviceSynchronize();
        cudaEventRecord(stop);
    } else {
        for(int i = 0; i < 3; i++) 
            if(sync_err[i] != cudaSuccess)
                std::cout << "CUDA error on " << i << "th allocation: " << cudaGetErrorString(sync_err[i]) << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double, std::milli>(end - begin);

    // ------------------------------ Error Handling ------------------------------

    // Runtime error from kernel execution
    async_err = cudaGetLastError();

    if(async_err != cudaSuccess) std::cout << "Async Error: " << cudaGetErrorString(async_err) << std::endl;
    for(int i = 0; i < 3; i++)
        if(sync_err[i] != cudaSuccess) std::cout << "Sync Error: " << cudaGetErrorString(sync_err[i]) << std::endl;

    // ------------------------------ Execution detailing ------------------------------

    // Time measuring via events
    cudaEventElapsedTime(&kernel_time, start, stop);
    std::cout << "(" << N << " data points, " << blocks << " blocks and " << cuda_threads << " threads per block)\n" 
                 "(" << total_time.count() << ", " << kernel_time << ")ms" << std::endl;

    // ------------------------------ Memory Freeing ------------------------------

    cudaFree(a); cudaFree(b); cudaFree(c);

    return 0;
}
