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

const size_t N = 8ULL*1024ULL * 1024ULL; // ULL = unsigned long long
const int BLOCK_SIZE = 256;

__global__ void reduce(float *gdata, float *out, size_t n){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = -INFINITY;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // grid stride loop to load the data
    // example: 640 (blocks) * 256 (threads per block) = 163,840 total threads
    //          thread 0 will compute data element 0, 163840, 327680, ...
    while (idx < n){
        sdata[tid] = fmaxf(sdata[tid], gdata[idx]);
        idx += gridDim.x * blockDim.x;
    }
    
    // parallel sweep reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        __syncthreads();
        if (tid < s) 
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main(){
    float *h_A, *h_sum, *d_A, *d_sums;
    const int blocks = 640;
    h_A = new float[N]; // allocate space for data in host memory
    h_sum = new float;
    float max_val = 5.0f;
    for (size_t i = 0; i < N; i++)
        h_A[i] = 1.0f;
    h_A[100] = max_val;
    cudaMalloc(&d_A, N*sizeof(float));
    cudaMalloc(&d_sums, blocks*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    
    // stage 1: block-level reduction. result is that block has it's own partial sum
    //          here we are using 640*256 threads simultaneously.
    reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
    cudaCheckErrors("reduction kernel launch failure");
    
    // stage 2: block reduction. reduce across blocks into single value d_A[0].
    //          
    reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks); 
    cudaCheckErrors("reduction kernel launch failure");
    
    cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");
    printf("output: %f, expected sum: %f, expected max %f\n", *h_sum, (float)((N-1) + max_val), max_val);
    return 0;
}