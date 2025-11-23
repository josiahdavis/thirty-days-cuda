#include <stdio.h>
#include <time.h>

/*

    -------  Matmul with Shared Memory  -------
    * Loading tiles of matrices A and B into a shared memory
    * Computing partial dot products using the cached data
    * Accumulating results across all tiles
    * Writing final results to global memory

    References:
        Exercise: https://github.com/olcf/cuda-training-series/tree/master/exercises/hw2
        Diagram: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=matrix%20multiply#shared-memory-matrix-multiplication-shared-memory
        Explanation: https://www.youtube.com/watch?v=3xfyiWhtvZw
*/

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

const int DSIZE = 8192;
const int block_size = 32; // cuda max is 1024 threads per block
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply C = A @ B
__global__ void matmul(const float *A, const float *B, float *C, int ds){
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if ((idx < ds) && (idy < ds)){
        float temp = 0;

        // Iterate through the tiles (chunks of data) to be processed. 
        for (int i = 0; i < ds/block_size; i++){
            /*
            Load tiles into shared memory
                Explanation of indexing parameters:
                A: 
                          idy * ds: Global row we will be working on (loop invariant)
                    i * block_size: New set of columns each iteration
                       threadIdx.x: The column within that set.

                B:
                     i * block_size * ds: New set of rows each iteration
                        threadIdx.y * ds: The row within that set
                                     idx: Global column we will be working on (loop invariant)
            */
            // general formula: row index * width + col index
            // getting col index is tricky: first get block index, then incrememnt by thread index x
            As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)]; 
            // getting row index is tricky: first get block index and then increment by thread index y
            Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

            // Every single thread within this block needs to be done before proceeding.
            __syncthreads();

            // Iterate within the tile.
            // This is a partial dot product with data in the shared memory.
            // We are sweeping through the rows within the A tile and cols within the B tile.
            for (int k = 0; k < block_size; k++)
                temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column

            // We don't want to overrite shared memory (with next tile) until all threads have completed their partial dot product.
            __syncthreads();
        }

        // Each thread is responsible for one element in the output matrix
        C[idy*ds + idx] = temp;
    }
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;
    t0 = clock();
    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE * DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }
    t1 = clock();
    t1sum = ((double)(t1-t0)/CLOCKS_PER_SEC);
    printf("Init took %f seconds. Begin compute \n", t1sum);

    // Copy input data to GPU
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Launch kernel
    dim3 block(block_size, block_size);
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
    matmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Copy results to host
    cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    t2 = clock();
    t2sum = ((double)(t2-t1)/CLOCKS_PER_SEC);
    printf("Done. Compute took %f seconds\n", t2sum);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0; i < DSIZE * DSIZE; i++){
        if (h_C[i] != A_val * B_val * DSIZE){
            printf("❌ Mismatch at index %d, was %f, should be %f\n", i, h_C[i], A_val * B_val * DSIZE);
            return -1;
        }
    }
    printf("✅ Success!\n");
    return 0;
}