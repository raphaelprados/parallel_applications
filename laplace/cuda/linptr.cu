// laplace_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define ITER_MAX 3000
#define CONV_THRESHOLD 1.0e-5

// CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Host matrices (for I/O / final save)
double *h_grid = NULL;

// GPU kernel:
// - compute new_grid[i,j] for interior cells
// - compute per-block max absolute difference and write it to blockMax[blockIdx]
__global__
void jacobi_kernel(const double *grid, double *new_grid, int N, double *blockMax) {
    extern __shared__ double sdata[]; // dynamic shared memory for reduction

    // 2D thread/block mapping
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;

    int i = by * bdy + ty; // row index
    int j = bx * bdx + tx; // col index

    // linear thread index inside block
    int tid = ty * bdx + tx;
    int blockSize = bdx * bdy;

    double local_max = 0.0;

    if (i >= 1 && i < N-1 && j >= 1 && j < N-1) {
        int idx = i * N + j;
        double up    = grid[(i-1)*N + j];
        double down  = grid[(i+1)*N + j];
        double left  = grid[i*N + (j-1)];
        double right = grid[i*N + (j+1)];

        double val = 0.25 * (left + right + up + down);
        new_grid[idx] = val;

        double diff = val - grid[idx];
        if (diff < 0.0) diff = -diff;
        local_max = diff;
    } else {
        // For threads outside the interior region, local_max stays 0
        local_max = 0.0;
    }

    // store local_max into shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Calculates the block's max error value. Uses a tree style reduction and works only for powers of 2. 
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double a = sdata[tid];
            double b = sdata[tid + stride];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // thread 0 writes block maximum to blockMax array
    if (tid == 0) {
        int blockId = by * gridDim.x + bx;
        blockMax[blockId] = sdata[0];
    }
}

// Utility to get wall-clock time (seconds)
double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

// Save host-side grid to file (same format as your original)
void save_grid_host(double *grid, int size) {
    char file_name[30];
    sprintf(file_name, "grid_laplace_cuda.txt");

    FILE *file = fopen(file_name, "w");
    if (!file) {
        fprintf(stderr, "Could not open file to save grid\n");
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(file, "%lf ", grid[i*size + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    
    // ###################################################################################################
    // ### Argument Handling    
    // ###################################################################################################
    
    if (argc != 2) {
        printf("Usage: ./laplace_cuda N\n");
        printf("N: The size of each side of the domain (grid)\n");
        return -1;
    }

    int N = 1 << atoi(argv[1]);
    if (N <= 2) {
        fprintf(stderr, "N must be > 2\n");
        return -1;
    }

    printf("Jacobi relaxation (CUDA) on %d x %d grid\n", N, N);

    // ###################################################################################################
    // ### Variable initialization    
    // ###################################################################################################

    size_t bytes = (size_t)N * N * sizeof(double);

    // allocate host grid and initialize
    h_grid = (double*)malloc(bytes);
    double *h_new = (double*)malloc(bytes);
    if (!h_grid || !h_new) {
        fprintf(stderr, "Host allocation failed\n");
        return -1;
    }

    // initialize grid: center heat region like original code
    int linf = N / 2;
    int lsup = linf + N / 10;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i >= linf && i < lsup && j >= linf && j < lsup)
                h_grid[i*N + j] = 100.0;
            else
                h_grid[i*N + j] = 0.0;
            h_new[i*N + j] = 0.0;
        }
    }

    // device allocations
    double *d_grid = NULL;
    double *d_new  = NULL;

    // ###################################################################################################
    // ### Memory allocation and data transfer
    // ###################################################################################################

    CUDA_CHECK(cudaMalloc((void**)&d_grid, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_new, bytes));

    // copy initial grid to device
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_new, bytes, cudaMemcpyHostToDevice));

    // ###################################################################################################
    // ### Kernel launch parameters 
    // ###################################################################################################

    // kernel launch parameters
    dim3 block(16, 16); // 256 threads per block
    dim3 gridDim( (N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y );

    int numBlocks = gridDim.x * gridDim.y;

    // allocate array of per-block maxima on device and host
    double *d_blockMax = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_blockMax, numBlocks * sizeof(double)));
    double *h_blockMax = (double*)malloc(numBlocks * sizeof(double));

    double err = 1.0;
    int iter = 0;

    double t0 = get_time_sec();

    // shared memory size: block.x * block.y doubles
    size_t shm_bytes = block.x * block.y * sizeof(double);

    // ###################################################################################################
    // ### Kernel launch/Main loop iterations    
    // ###################################################################################################

    while (err > CONV_THRESHOLD && iter <= ITER_MAX) {
        // launch kernel to compute new_grid and per-block max diffs
        jacobi_kernel<<<gridDim, block, shm_bytes>>>(d_grid, d_new, N, d_blockMax);
        CUDA_CHECK(cudaGetLastError());

        // copy per-block maxima back and reduce on host to get global err
        CUDA_CHECK(cudaMemcpy(h_blockMax, d_blockMax, numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
        double global_max = 0.0;
        for (int b = 0; b < numBlocks; ++b) {
            if (h_blockMax[b] > global_max) global_max = h_blockMax[b];
        }
        err = global_max;

        // swap device pointers for next iteration
        double *tmp = d_grid;
        d_grid = d_new;
        d_new = tmp;

        iter++;
    }

    double t1 = get_time_sec();

    // ###################################################################################################
    // ### Data saving and dealocation    
    // ###################################################################################################

    // copy final grid back to host (d_grid points to latest)
    CUDA_CHECK(cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost));

    // save result
    save_grid_host(h_grid, N);
    
    // cleanup
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_new));
    CUDA_CHECK(cudaFree(d_blockMax));
    free(h_grid);
    free(h_new);
    free(h_blockMax);

    printf("\nKernel executed in %lf seconds with %d iterations and error of %0.10lf\n",
           t1 - t0, iter, err);

    return 0;
}
