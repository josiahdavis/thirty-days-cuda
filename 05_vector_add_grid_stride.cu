/*
    --- vector add using a grid stride ---
    Blog post: https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    
    Vector add took 6.919403 seconds (blocks=1,threads=1)
    Vector add took 4.456653 seconds (blocks=1,threads=2)
    Vector add took 4.033181 seconds (blocks=1,threads=4)
    Vector add took 1.644261 seconds (blocks=1,threads=32)
    Vector add took 1.120367 seconds (blocks=1,threads=64)
    Vector add took 0.604691 seconds (blocks=1,threads=1024)
    Vector add took 0.616270 seconds (blocks=4,threads=256)

    --- background ----
    Video: https://vimeo.com/398824746

    Threads in a thread block (software) are not spread across multiple SMs (hardware)
    always only on a single SM. Shared memory is a per-SM resource.
    Warp - a collection of 32 threads.
    A thread block consists of 32-thread warps.
    A warp is executed physically in parallel (SIMD) on a mutliprocessor.
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

const int DSIZE = 32*1048576;

__global__ void vector_add(const float *A, const float *B, float *C, int ds){
    for (int idx = threadIdx.x + blockDim.x*blockIdx.x; idx < ds; idx += gridDim.x*blockDim.x){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    clock_t t0, t1;
    double t1sum=0.0;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];
    for (int i = 0; i < DSIZE; i++){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    t0 = clock();
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    int blocks = 1;
    int threads = 1024;
    // cuda max is 1024 threads per block. Will get below error if you try higher.
    // Fatal error: kernel launch failure (invalid argument at 05_vector_add_grid_stride.cu:67)
    // *** FAILED - ABORTING
    vector_add<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    t1 = clock();
    t1sum = ((double)(t1-t0)/CLOCKS_PER_SEC);
    printf("Vector add took %f seconds (blocks=%d,threads=%d)\n", t1sum, blocks, threads);
    cudaCheckErrors("kernel execution failure or cudaMemcpy D2H failure");
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}