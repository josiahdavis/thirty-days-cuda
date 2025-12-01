// References:
//  https://github.com/olcf/cuda-training-series/blob/master/exercises/hw6/array_inc.cu
//  Kernel execution takes 109505591 ns, previously it was taking 369731 ns.

#include <cstdio>
#include <cstdlib>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void inc(int *array, size_t n){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;  // Fixed: was blockDim.x instead of blockIdx.x
    while (idx < n){
        array[idx] += 1;
        idx += blockDim.x * gridDim.x;
    }
}

const size_t ds = 32ULL * 1024ULL * 1024ULL;

int main(){
    int *array;
    
    // Use unified memory allocation instead of regular malloc
    cudaMallocManaged(&array, ds*sizeof(array[0]));
    cudaCheckErrors("cudaMallocManaged error");
    
    // Initialize memory (memset works with unified memory)
    memset(array, 0, ds*sizeof(array[0]));
    
    // Prefetch data to GPU (device 0)
    int device = 0;
    cudaGetDevice(&device);
    cudaMemLocation deviceLoc = {};
    deviceLoc.type = cudaMemLocationTypeDevice;
    deviceLoc.id = device;
    cudaMemPrefetchAsync(array, ds*sizeof(array[0]), deviceLoc, 0);
    cudaCheckErrors("cudaMemPrefetchAsync to GPU error");
    
    inc<<<256,256>>>(array, ds);
    cudaCheckErrors("kernel launch failure");
    
    // Prefetch data back to CPU
    cudaMemLocation cpuLoc = {};
    cpuLoc.type = cudaMemLocationTypeHost;
    cpuLoc.id = 0;
    cudaMemPrefetchAsync(array, ds*sizeof(array[0]), cpuLoc, 0);
    cudaCheckErrors("cudaMemPrefetchAsync to CPU error");
    
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel execution or synchronization error");
    
    for (int i = 0; i < ds; i++){
        if (array[i] != 1){
            printf("Mismatch at %d, was %d, expected %d\n", i, array[i], 1);
            return 1;
        }
    }
    printf("Success!!\n");
    
    // Clean up
    cudaFree(array);
    
    return 0;
}
