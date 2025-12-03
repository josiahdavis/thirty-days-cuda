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

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes){
    ptr = (T)malloc(num_bytes);
}

__global__ void inc(int *array, size_t n){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    // Grid stride loop
    while (idx < n){
        array[idx] += 1;
        idx += blockDim.x * gridDim.x;
    }
}

const size_t ds = 32ULL * 1024ULL * 1024ULL;

int main(){
    int *array;
    cudaMallocManaged(&array, ds*sizeof(array[0]));
    cudaCheckErrors("malloc error");
    for (size_t i = 0; i < ds; i++)
        array[i] = 2;
    inc<<<256,256>>>(array, ds);
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel executioin or cudaMemcpy D2H Error");
    for (int i = 0; i < ds; i++){
        if (array[i] != 3){
            printf("Mismach at %d, was %d, expected %d\n", i, array[i], 3);
            return -1;
        }
    }
    printf("Success!\n");
    return 0;
}