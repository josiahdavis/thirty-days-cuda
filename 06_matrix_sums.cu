/*
    --- Notes on Memory Layout ---
    Local storage:
        Each thread has it's own local storage. 
        Most important resource here is registers.
        Generally managed by the compiler (not the programmer).
    Shared memory and the L1 cache: 
        Accessible by threads in the same threadblock.
        Shared memory can be managed by the user/programmer.
        L1 cache is not visible to the user/programmer.
        Very high throughput and low latency.
    L2 cache: 
        Accessible across all blocks on the device. 
        All accesses to global memory go through L2.
    Global memory (e.g., 40GB for an A100)
        Accessible by all threads as well as host (CPU).
        High latency (100s of cycles)
            (the difference between when you ask for something and when you recieve it)
*/

#include <stdio.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const size_t DSIZE = 16384; // matrix side dim
const int block_size = 256; // cuda max is 1024

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < ds){
        float sum = 0.0f;
        for (size_t i = 0; i < ds; i++){
            sum += A[idx * ds + i];
        }
    sums[idx] = sum;
    }
}

// matrix column sum kernel 
__global__ void column_sums(const float *A, float *sums, size_t ds){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds){
        float sum = 0.0f;
        for (size_t i = 0; i < ds; i++)
            sum += A[i * ds + idx];
    sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz){
    for (size_t i = 0; i < sz; i++){
        if (data[i] != (float)sz){
            printf("results mismatch at %lu, was %f, should be %f\n", i, data[i], (float)sz);
            return false;
        }
    }
    return true;
}

int main(){

    // Allocated space on host and initialize
    float *h_A, *h_sums, *d_A, *d_sums;
    h_A = new float[DSIZE*DSIZE]; // allocate space for data in host memory
    h_sums = new float[DSIZE]();
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = 1.0f;
    }

    // Allocate space on device
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_sums, DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure.");

    // Copy matrix A to device
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Cuda processing sequence step 1 is complete
    row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure.");

    // Copy vector sums from device to host
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE)) return -1;
    printf("✅ Row sums correct!\n");

    cudaMemset(d_sums, 0, DSIZE*sizeof(float));
    column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure.");

    // Copy vector sums device to host
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE)) return -1;
    printf("✅ Column sums correct!\n");
    return 0;
}