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

// Block-level tiling dimensions
#define BLOCK_TILE_M 128    // Number of rows per thread block (was BM)
#define BLOCK_TILE_N 128    // Number of columns per thread block (was BN)
#define BLOCK_TILE_K 32     // K-dimension of shared memory tiles (was BK)

// Thread-level sub-tiling dimensions (register blocking)
#define THREAD_TILE_M 8     // Rows computed per thread (was TM)
#define THREAD_TILE_N 8     // Columns computed per thread (was TN)

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
 * Method 1: Shared Memory Matmul with 2D blocktiling (84% speed of cuBLAS.)
 * Computes: C = A * B where A is (M x K) and B is (K x N)
 * Modified from Philip Fabianek's kernel: 
 *    https://github.com/philipfabianek/cuda-gemm-from-scratch/blob/main/src/kernels/04_2D_coarsened.cuh
 */

__global__ void customMatMul(float* A, float* B, float* C, int M, int K, int N) {
    
    // ===== STEP 1: SETUP THREAD BLOCK POSITIONING =====
    
    // Calculate the starting position for this thread block's tile in global memory
    // Each block processes a BLOCK_TILE_M x BLOCK_TILE_N section of matrix C
    float* block_A_start = A + (blockIdx.y * BLOCK_TILE_M * K);           // Start of A tile rows
    float* block_B_start = B + (blockIdx.x * BLOCK_TILE_N);               // Start of B tile columns  
    float* block_C_start = C + (blockIdx.y * BLOCK_TILE_M * N) + (blockIdx.x * BLOCK_TILE_N);
    
    // ===== STEP 2: ALLOCATE SHARED MEMORY TILES =====
    
    // Shared memory tiles for cooperative loading and reuse across threads
    __shared__ float shared_A[BLOCK_TILE_M * BLOCK_TILE_K];   // Cache A tile in shared memory
    __shared__ float shared_B[BLOCK_TILE_K * BLOCK_TILE_N];   // Cache B tile in shared memory
    
    // ===== STEP 3: CALCULATE THREAD-TO-MEMORY MAPPING =====
    
    // Map thread ID to position for loading data into shared memory
    // This mapping ensures coalesced global memory accesses (consecutive threads access consecutive memory)
    const int threads_per_block = (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N);
    
    // For loading matrix A into shared memory
    const int load_A_col = threadIdx.x % BLOCK_TILE_K;              // Column in A tile (0,1,2,...,31,0,1,2,...)
    const int load_A_row = threadIdx.x / BLOCK_TILE_K;              // Row in A tile (0,0,0,...,1,1,1,...)
    const int load_A_stride = threads_per_block / BLOCK_TILE_K;     // Rows per iteration when loading A
    
    // For loading matrix B into shared memory  
    const int load_B_col = threadIdx.x % BLOCK_TILE_N;              // Column in B tile
    const int load_B_row = threadIdx.x / BLOCK_TILE_N;              // Row in B tile
    const int load_B_stride = threads_per_block / BLOCK_TILE_N;     // Rows per iteration when loading B
    
    // ===== STEP 4: CALCULATE THREAD'S OUTPUT POSITION =====
    
    // Each thread is responsible for computing a THREAD_TILE_M x THREAD_TILE_N sub-tile of C
    const int threads_per_row = BLOCK_TILE_N / THREAD_TILE_N;       // Number of thread tiles per row
    const int output_col_start = (threadIdx.x % threads_per_row) * THREAD_TILE_N;  // Starting column
    const int output_row_start = (threadIdx.x / threads_per_row) * THREAD_TILE_M;  // Starting row
    
    // ===== STEP 5: ALLOCATE REGISTERS FOR COMPUTATION =====
    
    // Accumulator registers - store partial sums for this thread's output sub-tile
    float thread_results[THREAD_TILE_M * THREAD_TILE_N] = {0.0f};
    
    // Register arrays for reused values during inner loop computation
    float A_registers[THREAD_TILE_M] = {0.0f};    // Cache A values for reuse
    float B_registers[THREAD_TILE_N] = {0.0f};    // Cache B values for reuse
    
    // ===== STEP 6: MAIN COMPUTATION LOOP - ITERATE OVER K DIMENSION =====
    
    // Process the matrix multiplication in chunks of BLOCK_TILE_K
    for (int k_block = 0; k_block < K; k_block += BLOCK_TILE_K) {
        
        // ----- STEP 6a: COOPERATIVELY LOAD DATA INTO SHARED MEMORY -----
        
        // Load A tile from global memory to shared memory with coalesced access
        // Each thread may load multiple elements to fill the entire tile
        for (int row_offset = 0; row_offset < BLOCK_TILE_M; row_offset += load_A_stride) {
            int global_row = load_A_row + row_offset;
            int global_col = load_A_col;
            int shared_idx = (global_row * BLOCK_TILE_K) + global_col;
            int global_idx = (global_row * K) + global_col;
            
            shared_A[shared_idx] = block_A_start[global_idx];
        }
        
        // Load B tile from global memory to shared memory with coalesced access
        for (int row_offset = 0; row_offset < BLOCK_TILE_K; row_offset += load_B_stride) {
            int global_row = load_B_row + row_offset;
            int global_col = load_B_col;
            int shared_idx = (global_row * BLOCK_TILE_N) + global_col;
            int global_idx = (global_row * N) + global_col;
            
            shared_B[shared_idx] = block_B_start[global_idx];
        }
        
        // Wait for all threads to finish loading before proceeding
        __syncthreads();
        
        // ----- STEP 6b: ADVANCE TO NEXT K-BLOCK FOR NEXT ITERATION -----
        block_A_start += BLOCK_TILE_K;              // Move to next K-block in A
        block_B_start += BLOCK_TILE_K * N;          // Move to next K-block in B
        
        // ----- STEP 6c: COMPUTE USING SHARED MEMORY DATA -----
        
        // For each element in the K dimension of our shared memory tiles
        for (int k_elem = 0; k_elem < BLOCK_TILE_K; k_elem++) {
            
            // Load this thread's slice of A values into registers
            // (vertical slice: THREAD_TILE_M consecutive elements from column k_elem)
            for (int i = 0; i < THREAD_TILE_M; i++) {
                int shared_A_idx = ((output_row_start + i) * BLOCK_TILE_K) + k_elem;
                A_registers[i] = shared_A[shared_A_idx];
            }
            
            // Load this thread's slice of B values into registers  
            // (horizontal slice: THREAD_TILE_N consecutive elements from row k_elem)
            for (int j = 0; j < THREAD_TILE_N; j++) {
                int shared_B_idx = (k_elem * BLOCK_TILE_N) + (output_col_start + j);
                B_registers[j] = shared_B[shared_B_idx];
            }
            
            // Compute outer product: A_slice ⊗ B_slice and accumulate results
            // This produces THREAD_TILE_M x THREAD_TILE_N partial products
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    int result_idx = (i * THREAD_TILE_N) + j;
                    thread_results[result_idx] += A_registers[i] * B_registers[j];
                }
            }
        }
        
        // Synchronize before loading next tile (prevent read-after-write hazards)
        __syncthreads();
    }
    
    // ===== STEP 7: WRITE RESULTS TO GLOBAL MEMORY =====
    
    // Each thread writes its THREAD_TILE_M x THREAD_TILE_N sub-tile to global memory C
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int result_idx = (i * THREAD_TILE_N) + j;
            int global_row = output_row_start + i;
            int global_col = output_col_start + j;
            int global_idx = (global_row * N) + global_col;
            
            block_C_start[global_idx] = thread_results[result_idx];
        }
    }
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
    
    // int M = 2048;
    // int K = 2048;
    // int N = 2048;
    
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
    
    dim3 gridDim((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    dim3 blockDim((BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N));
    
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
