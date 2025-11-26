#include <stdio.h>

/*
    --- Sum Reduction using Warp Shuffle ---
    * A warp = set of 32 threads.
    * Can communicate thread <-> thread within the warp without using memory of any kind.
    
    --- References ---
    * Modified from this NVIDIA example: https://github.com/olcf/cuda-training-series/blob/master/exercises/hw5/reductions.cu
    * Also see CUDA Atomics, Reductions, and Warp Shuffle lecture: https://vimeo.com/419029739
*/


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

__global__ void reduce_warp_shuffle(float *gdata, float *out){
    __shared__ float sdata[32]; // Why sdata[32]? 
                                // max warps in a thread block is 1024 (threads per block) / 32 (threads per warp) = 32
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float val = 0.0f;                    // Each thread has it's own independent running sum. Will accumulate using grid stride loop.
    unsigned mask = 0xFFFFFFFFU;         // Specifies a 1 for each of the 32 bits, meaning we need to use all 32 threads in the warp.
    int lane = threadIdx.x % warpSize;   // Lane = Which thread am I in the warp? threadIdx.x \in (0, thread block size)
    int warpID = threadIdx.x / warpSize; // warpID = Which warp am I in?

    // Grid stride loop to load
    while (idx < N){
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }
    /*
        1st warp shuffle reduction:
        Each warp is going to create it's own partial sum
    */
    for (int offset = warpSize/2; offset > 0; offset >>=1 ){
        /*

            offset = 16:
                thread 0 will take it's val and add val from thread 16
                thread 1 will take it's val and add val from thread 17
                ....
            offset = 8:
                thread 0 will take it's val and add val from thread 8
                thread 1 will take it's val and add val from thread 9
                ....
            ....
        */
        val += __shfl_down_sync(mask, val, offset);
    }
    // lane 0 is the thread in the warp responsible for accumulating the partial sum
    if (lane == 0) sdata[warpID] = val;

    // Need to make sure every warp has finished it's partial sum before proceeding.
    __syncthreads();

    // Hereafter we are just processing warp 0
    if (warpID == 0){
        // Reload val from shared memory if warp existed
        val = (tid < blockDim.x/warpSize) ? sdata[lane] : 0;
        for (int offset = warpSize/2; offset > 0; offset >>=1){
            val += __shfl_down_sync(mask, val, offset);
        }
        // At this point val is partial sum for the entire thread block. 
        // Last step is to update our global value across thread blocks.
        if (tid == 0) atomicAdd(out, val);
    }
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
  const int blocks = 640;
  reduce_warp_shuffle<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); return -1;}
  printf("✅ Reduction warp shuffle sum correct!\n");
  return 0;
}