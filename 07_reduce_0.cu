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

// naive atomic reduction kernel
__global__ void atomic_red(const float *gdata, float *out){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) atomicAdd(out, gdata[idx]);
}

// reduction w/atomic sum
__global__ void reduce_a(float *gdata, float *out){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Stride across grid to load the data
    while (idx < N){
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    // parallel sweep reduction
    // s>>=1 means bit shift right by 1, i.e., divide by two.
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        __syncthreads();
        if (tid < s) 
            sdata[tid] += sdata[tid + s];
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

int main(){
  float *h_A, *h_sum, *d_A, *d_sum;
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  for (int i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, N*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sum, sizeof(float));  // allocate device space for sum
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); return -1;}
  printf("atomic sum reduction correct!\n");
  const int blocks = 640;
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction w/atomic sum incorrect!\n"); return -1;}
  printf("reduction w/atomic sum correct!\n");
  return 0;
}