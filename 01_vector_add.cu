#include <stdio.h>

// References: 
//      ORNL exercise: https://github.com/olcf/cuda-training-series/blob/master/exercises/hw1/vector_add.cu


// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int DSIZE = 4096;
const int block_size = 256; // CUDA max is 1024

// vector add kernel: C = A + B
__global__ void vector_add(const float *A, const float *B, float *C, int ds){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // create 1D thread index from built-in variable
    if (idx < ds){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE]; // allocate space for vectors in host memory
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++){ // initialize vectors in host memory
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }

    // STEP 1: Copy data to device
    cudaMalloc(&d_A, DSIZE*sizeof(float)); // allocate space on device for vectors
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));

    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice); // copy vector A to device
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // STEP 2: Run CUDA Kernel
    vector_add<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE); // number of blocks uses ceiling division
    cudaCheckErrors("kernel launch failure");

    // STEP 3: Copy vector C from device to host
    cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy host-to-device failure");
    printf("A[3] = %f\n", h_A[3]);
    printf("B[3] = %f\n", h_B[3]);
    printf("C[3] = %f\n", h_C[3]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); // clean up memory
    free(h_A); free(h_B); free(h_C);
    return 0;
}