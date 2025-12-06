/*
 * Comparison of Naive Matrix Multiplication Shared Memory with CUBLAS.
 * Compile: nvcc 17_matmul_shared.cu -o main -lcublas
 * Run: ./main
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32 // cuda max is 1024 threads per block

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * Method 1: Naive Matrix Multiplication Kernel
 * Computes: C = A * B where A is (M x K) and B is (K x N)
 */
__global__ void naiveMatMul(float* A, float* B, float* C, int M, int K, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if ((idy < M) && (idx < N)){
        float temp = 0;
        for (int i = 0; i < K; i++){
            temp += A[idy * K + i] * B[i * N + idx];
        }
        C[idy * N + idx] = temp;
    }
}

/*
 * Method 2: Shared Memory Matrix Multiplication Kernel
 * Computes: C = A * B where A is (M x K) and B is (K x N)
 */
__global__ void sharedMatMul(float* A, float* B, float* C, int M, int K, int N) {
    // Note 1: formula for indexing into a matrix to get correct element:
    //      row_index * width + col_index
    // Note 2: using block and tile terms interchangeably. Since what we are sharing within is threads in a block.

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x; // blockDim.x == blockDim.y == BLOCK_SIZE
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    
    float temp = 0;
    
    // Loop over cache blocks aka "tiles"
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++){
        // Step 1: Load A tile and B tile into shared memory
        // Load tile A into shared memory
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (idy < M && aCol < K){
            As[threadIdx.y][threadIdx.x] = A[idy * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load tile B into shared memory
        int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (idx < N && bRow < K){
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Sync within the block to ensure all values within the threadblock are loaded.
        __syncthreads();
        // At this point As and Bs are fully populated with the appropriate data 
        // Reminder all threads execute in parallel for given t.

        // Step 2: Compute partial dot product for this tile only using shared memory
        for (int k = 0; k < BLOCK_SIZE; k++){
            temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Sync threads across block before loading next tile
        __syncthreads();
    }
    
    // Final step to be complete only after going through each tile
    // Write results to global memory. 
    // Each thread has a single result to contribute.
    if (idy < M && idx < N){
        C[idy * N + idx] = temp;
    }
}

/*
 * Method 3: cuBLAS Matrix Multiplication
 * cuBLAS uses column-major order, so we compute C = B^T * A^T to get row-major result.
 * This gives C = A * B in row-major format.
 */
void matMulCuBLAS(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int M, int K, int N){
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS does: C = alpha * op(A) * op(B) + beta * C
    // We want row-major C = A * B
    // In column-major this is C^T = B^T * A^T
    // So we swap the order and use CUBLAS_OP_N (no transpose)
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transpose
        N, M, K, // Dimensions swapped for column-major
        &alpha,
        d_B, N,   // B is N x K in column-major
        d_A, K,   // A is K x M in column-major
        &beta,
        d_C, N    // C is N x M in column-major
    ));
}


/*
 * CPU reference implementation for verification
 */
void matMulCPU(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * Initialize matrix with random values
 */
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;  // Random values 0-10
    }
}

/*
 * Verify GPU results against CPU
 */
bool verifyResults(float* gpu, float* cpu, int size, float epsilon = 1e-1) {
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > epsilon) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                   i, gpu[i], cpu[i], diff);
            return false;
        }
    }
    return true;
}

/*
 * Print a small portion of matrix for debugging
 */
void printMatrix(const char* name, float* mat, int rows, int cols, int maxPrint = 4) {
    printf("%s (%dx%d):\n", name, rows, cols);
    int printRows = (rows < maxPrint) ? rows : maxPrint;
    int printCols = (cols < maxPrint) ? cols : maxPrint;
    
    for (int i = 0; i < printRows; i++) {
        for (int j = 0; j < printCols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        if (cols > maxPrint) printf("...");
        printf("\n");
    }
    if (rows > maxPrint) printf("...\n");
    printf("\n");
}

int main() {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    // int M = 1024;
    // int K = 512;
    // int N = 128;

    int M = 4096;
    int K = 4096;
    int N = 4096;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║    Matrix Multiplication: Naive vs cuBLAS vs CPU               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    printf("Computing C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    printf("Total FLOPs: %.2f billion\n\n", 2.0 * M * N * K / 1e9);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_naive = (float*)malloc(sizeC);
    float *h_C_shared = (float*)malloc(sizeC);
    float *h_C_cublas = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);
    
    // Initialize matrices
    srand(42);
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Create library handles
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsedTime;
    
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("METHOD 1: Naive Matmul Kernel\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Warm-up
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    double gflops1 = (2.0 * M * N * K) / (elapsedTime * 1e6);
    printf("Time:        %.3f ms\n", elapsedTime);
    printf("Performance: %.2f GFLOPS\n\n", gflops1);

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("METHOD 2: Shared Matmul Kernel\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
    // Warm-up
    sharedMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    sharedMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_shared, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    double gflops2 = (2.0 * M * N * K) / (elapsedTime * 1e6);
    printf("Time:        %.3f ms\n", elapsedTime);
    printf("Performance: %.2f GFLOPS\n\n", gflops2);
    
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("METHOD 3: cuBLAS (cublasSgemm)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // Warm-up
    matMulCuBLAS(cublasHandle, d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    matMulCuBLAS(cublasHandle, d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    double gflops3 = (2.0 * M * N * K) / (elapsedTime * 1e6);
    printf("Time:        %.3f ms\n", elapsedTime);
    printf("Performance: %.2f GFLOPS\n", gflops3);
    printf("Speedup vs Naive: %.2fx\n", gflops3 / gflops1);
    printf("Naive relative cuBLAS: %.1f%%\n\n", gflops1 / gflops3 * 100);
    printf("Shared relative cuBLAS: %.1f%%\n\n", gflops2 / gflops3 * 100);
    
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("CPU Reference (single-threaded)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    auto cpuStart = clock();
    matMulCPU(h_A, h_B, h_C_ref, M, K, N);
    auto cpuEnd = clock();
    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    double cpuGflops = (2.0 * M * N * K) / (cpuTime * 1e6);
    
    printf("Time:        %.3f ms\n", cpuTime);
    printf("Performance: %.2f GFLOPS\n\n", cpuGflops);

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Verification\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    printf("Naive Kernel vs CPU: ");
    if (verifyResults(h_C_naive, h_C_ref, M * N)) {
        printf("✅ PASS\n");
    } else {
        printf("❌ FAIL\n");
    }

    printf("Shared Kernel vs CPU: ");
    if (verifyResults(h_C_shared, h_C_ref, M * N)) {
        printf("✅ PASS\n");
    } else {
        printf("❌ FAIL\n");
    }
    
    printf("cuBLAS vs CPU:       ");
    if (verifyResults(h_C_cublas, h_C_ref, M * N)) {
        printf("✅ PASS\n");
    } else {
        printf("❌ FAIL\n");
    }
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                         SUMMARY                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("Naive Kernel:   %.2f GFLOPS (%.0fx faster than CPU)\n", gflops1, gflops1/cpuGflops);
    printf("Naive Kernel:   %.2f GFLOPS (%.0fx faster than CPU)\n", gflops2, gflops2/cpuGflops);
    printf("cuBLAS:         %.2f GFLOPS (%.0fx faster than CPU)\n", gflops3, gflops3/cpuGflops);
    printf("\nKey Takeaway: cuBLAS is %.1fx faster than the naive kernel!\n", gflops2/gflops1);
    printf("                Naive kernel is %.1f%% the speed of cuBLAS.\n", gflops1/gflops3*100);
    printf("                Shared kernel is %.1f%% the speed of cuBLAS.\n", gflops2/gflops3*100);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    free(h_C_cublas);
    free(h_C_ref);
    return 0;
}