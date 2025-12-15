/*
 * Comparison of custom Matrix Multiplication with CUBLAS.
 * Compile: nvcc 24_matmul.cu -o main -lcublas
 * Run: ./main
 */

#include <cassert>
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
 * Method 1: Global Memory Coalescing (13.5% – 17.3% speed of cuBLAS.)
 * Computes: C = A * B where A is (M x K) and B is (K x N)
 * Source: https://siboehm.com/articles/22/CUDA-MMM (Kernel #2)
 */

__global__ void customMatMul(float* A, float* B, float* C, int M, int K, int N) {
    // Original approach: 
    //      1.5%  speed of cuBLAS if I use row=x, col=y
    //      13.5% speed of cuBLAS if I use col=x, row=y (I even got up to 17.3%!) I think this is going what Simon was doing already but in a more simple/idiomatic way! 
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
 
    if (y < M && x < N){
        float tmp = 0.0;
        for (int i = 0; i < K; i++){
            tmp += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = tmp;
    }

    // // Simon's approach
    // //      13.8% speed of cuBLAS, so basically the same as if I just switch x and y and do things the idiomatic way.
    // const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE); // row
    // const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE); // col
    
    // if (x < M && y < N){
    //     float tmp = 0.0;
    //     for (int i = 0; i < K; i++){
    //         tmp += A[x * K + i] * B[i * N + y];
    //     }
    //     C[x * N + y] = tmp;
    // }
}

/*
 * Method 2: cuBLAS Matrix Multiplication
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
            printf("Mismatch at index %d: Custom=%.4f, cuBLAS=%.4f, diff=%.4f\n",
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

    // int M = 256;
    // int K = 256;
    // int N = 256;

    // int M = 1024;
    // int K = 256;
    // int N = 128;

    int M = 4096;
    int K = 4096;
    int N = 4096;
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║    Matrix Multiplication: custom vs cuBLAS vs CPU               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    printf("Computing C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    printf("Total FLOPs: %.2f billion\n\n", 2.0 * M * N * K / 1e9);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_custom = (float*)malloc(sizeC);
    float *h_C_cublas = (float*)malloc(sizeC);
    
    // Initialize matrices
    srand(42);
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C2;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    CUDA_CHECK(cudaMalloc(&d_C2, sizeC));
    
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
    printf("METHOD 1: custom Matmul Kernel\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    // Warm-up
    customMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    customMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_custom, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    double gflops1 = (2.0 * M * N * K) / (elapsedTime * 1e6);
    printf("Time:        %.3f ms\n", elapsedTime);
    printf("Performance: %.2f GFLOPS\n\n", gflops1);
    
    printMatrix("Matrix A", h_A, M, K);
    printMatrix("Matrix B", h_B, K, N);
    printMatrix("Matrix C", h_C_custom, M, N);

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("METHOD 2: cuBLAS (cublasSgemm)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // Warm-up
    matMulCuBLAS(cublasHandle, d_A, d_B, d_C2, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    matMulCuBLAS(cublasHandle, d_A, d_B, d_C2, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C2, sizeC, cudaMemcpyDeviceToHost));
    
    double gflops2 = (2.0 * M * N * K) / (elapsedTime * 1e6);
    printf("Time:        %.3f ms\n", elapsedTime);
    printf("Performance: %.2f GFLOPS\n", gflops2);
    printf("Speedup vs custom: %.2fx\n", gflops2 / gflops1);
    printf("custom relative cuBLAS: %.1f%%\n\n", gflops1 / gflops2 * 100);

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Verification\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    printf("custom Kernel vs cuBLAS: ");
    if (verifyResults(h_C_custom, h_C_cublas, M * N)) {
        printf("✅ PASS\n");
    } else {
        printf("❌ FAIL\n");
    }
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                         SUMMARY                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("custom Kernel:   %.2f GFLOPS\n", gflops1);
    printf("cuBLAS:         %.2f GFLOPS\n", gflops2);
    printf("\nKey Takeaway: cuBLAS is %.1fx faster than the custom kernel!\n", gflops2/gflops1);
    printf("                custom kernel is %.1f%% the speed of cuBLAS.\n", gflops1/gflops2*100);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C2));
    free(h_A);
    free(h_B);
    free(h_C_custom);
    free(h_C_cublas);
    return 0;
}