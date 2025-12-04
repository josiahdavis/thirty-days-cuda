/*
 * Complete Tiled Matrix Multiplication Example
 * Compile: nvcc -o matmul matmul_tiled.cu
 * Run: ./matmul matmul_tiled
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * Tiled Matrix Multiplication Kernel with Shared Memory
 * Computes: C = A * B where A is (M x K) and B is (K x N)
 */
__global__ void tiledMatMul(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++){
        // load tile from A
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K){
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // load tile from B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N){
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++){
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();

    }

    if (row < M && col < N){
        C[row * N + col] = sum;
    }
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
bool verifyResults(float* gpu, float* cpu, int size, float epsilon = 1e-2) {
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
    int M = 1024;   // Rows of A and C
    int K = 256;   // Cols of A, Rows of B
    int N = 512;   // Cols of B and C
    
    printf("=== CUDA Tiled Matrix Multiplication ===\n");
    printf("Computing C(%dx%d) = A(%dx%d) * B(%dx%d)\n\n", M, N, M, K, K, N);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);  // CPU reference result
    
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    srand(42);  // Fixed seed for reproducibility
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    printMatrix("Matrix A", h_A, M, K);
    printMatrix("Matrix B", h_B, K, N);
    
    // Allocate device memory
    printf("Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    // Setup kernel launch parameters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Launch config: Grid(%d,%d), Block(%d,%d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Warm-up run
    tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Timed GPU execution
    printf("\nRunning GPU kernel...\n");
    CUDA_CHECK(cudaEventRecord(start));
    tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    
    // Copy result back to host
    printf("Copying result back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // CPU reference computation
    printf("Running CPU reference...\n");
    auto cpuStart = clock();
    matMulCPU(h_A, h_B, h_C_ref, M, K, N);
    auto cpuEnd = clock();
    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
    
    // Verify results
    printf("\nVerifying results...\n");
    bool correct = verifyResults(h_C, h_C_ref, M * N);
    
    if (correct) {
        printf("✅ Results match! GPU computation is correct.\n\n");
    } else {
        printf("❌ Results do NOT match!\n\n");
    }
    
    printMatrix("Result C (GPU)", h_C, M, N);
    
    // Performance metrics
    double gflops = (2.0 * M * N * K) / (gpuTime * 1e6);  // Billions of FLOPs
    
    printf("=== Performance ===\n");
    printf("GPU Time: %.3f ms\n", gpuTime);
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Speedup:  %.2fx\n", cpuTime / gpuTime);
    printf("GPU Performance: %.2f GFLOPS\n", gflops);
    
    // Memory bandwidth
    double bytesTransferred = (sizeA + sizeB + sizeC);
    double bandwidthGB = bytesTransferred / (gpuTime * 1e6);  // GB/s
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidthGB);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    printf("\nDone! Next up, compare to cuBLAS!\n");
    return 0;
}