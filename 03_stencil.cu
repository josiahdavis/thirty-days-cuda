#include <stdio.h>
#include <algorithm>
/*
    -- Shared memory --
    A per-block resource: Each block has it's own local copy of shared memory.
    It's like a user-managed cache.

    -- This kernel -- 
    Reads block width + 2 * radius input elements from global memory into shared memory.
    Computes block width output elements.
    Writes block width output elements from shared memory to global memory.
*/

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int global_index = threadIdx.x + blockIdx.x * blockDim.x; // same approach as vector_add
    int left_index = threadIdx.x + RADIUS; // index within the shared memory

    // Read input elements into shared memory
    temp[left_index] = in[global_index]; // Gives us everything except the halo
    if (threadIdx.x < RADIUS){
        temp[left_index - RADIUS] = in[global_index - RADIUS];
        temp[left_index + BLOCK_SIZE] = in[global_index + BLOCK_SIZE];
    }

    // All threads must have loaded the shared memory before we can begin our stencil operation
    // Common pattern: loading of data in shared memory, use of data in shared memory
    __syncthreads();

    // Now each thread is responsible for reading 7 elements out 
    // of our input dataset which has been cached in shared memory
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[left_index + offset];
    out[global_index] = result;
}

void fill_ints(int *x, int n){
    fill_n(x, n, 1);
}

int main(void){
    int *in, *out; 
    int *d_in, *d_out;

    int size = (N+2*RADIUS) * sizeof(int);
    in = (int *)malloc(size); fill_ints(in, N + 2*RADIUS);
    out = (int *)malloc(size); fill_ints(out, N + 2*RADIUS);
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // STEP 1: Copy from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // STEP 2: Launch kernel on GPU
    stencil_1d<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);

    // STEP 3: Copy back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Error checking
    for (int i = 0; i < N + 2*RADIUS; i++){
        if (i < RADIUS || i >= N+RADIUS){
            if (out[i] != 1)
                printf("❌ Mismatch at index %d, was %d, should be: %d\n", i, out[i], 1);
        } else {
            if (out[i] != 1 + 2 * RADIUS)
                printf("❌ Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2*RADIUS);
        }
    }

    // Cleanup 
    free(in); free(out);
    cudaFree(d_in); cudaFree(d_out);
    printf("✅ Success!\n");
}