#include <stdio.h>

#define TILE_SIZE 16

__global__ void transpose_naive(float* input, float* output, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        output[col * height + row] = input[row * width + col];
    }
}

__global__ void transpose_shared_memory(float* input, float *output, int width, int height){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (col < width && row < height){
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    __syncthreads();

    int out_col = blockIdx.y * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (out_col < height && out_row < width){
        output[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}


void init_matrix(float* matrix, int width, int height){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            matrix[i * width + j] = i * width + j; // sequential numbers
        }
    }
}

bool verify_transpose(float* original, float* transposed, int width, int height){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            if (original[i * width + j] != transposed[j * height + i]) {
                printf("Mismatch at (%d, %d): expected %f, got %f\n", i, j, 
                original[i * width + j], transposed[j * height + i]);
                return false;
            }
        }
    }
    return true;
}

int main(){
    const int WIDTH = 1024;
    const int HEIGHT = 512;
    const int size = WIDTH * HEIGHT * sizeof(float);

    float *h_input = (float*)(malloc(size));
    float *h_output_naive = (float*)(malloc(size));
    float *h_output_shared = (float*)(malloc(size));
    init_matrix(h_input, WIDTH, HEIGHT);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    int block_size = 16;
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((WIDTH + block_size - 1) / block_size, (HEIGHT + block_size - 1) / block_size);
    printf("Matrix size: %dx%d\n", WIDTH, HEIGHT);
    printf("Block size: %dx%d\n", blockSize.x, blockSize.y);
    printf("Grid size: %dx%d\n", gridSize.x, gridSize.y);


    printf("\nTesting naive transpose...\n");
    transpose_naive<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_naive, d_output, size, cudaMemcpyDeviceToHost);
    
    if (verify_transpose(h_input, h_output_naive, WIDTH, HEIGHT)) {
        printf("✅ Naive transpose: PASSED\n");
    } else {
        printf("❌ Naive transpose: FAILED\n");
    }

    printf("\nTesting shared memory transpose...\n");
    transpose_shared_memory<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_shared, d_output, size, cudaMemcpyDeviceToHost);

    if (verify_transpose(h_input, h_output_shared, WIDTH, HEIGHT)) {
        printf("✅ Shared memory transpose: PASSED\n");
    } else {
        printf("❌ Shared memory transpose: FAILED\n");
    }

    // Cleanup
    free(h_input);
    free(h_output_naive); 
    free(h_output_shared); 
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\nNext up: Use nvprof or Nsight Compute to examine memory access patterns!\n");
    return 0;
}