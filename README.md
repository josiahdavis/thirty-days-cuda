# thirty-days-cuda

* Write one CUDA kernel for thirty days. 
* Each kernel is fully self-contained in a single file.
* Here is an example of how to run: 

```
nvcc -o main 00_hello.cu
./main
```

* Here is example output:

```
Hello from block 1, thread 0
Hello from block 1, thread 1
Hello from block 1, thread 2
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
```

## Resources

* Oak Ridge National Lab hosted a great NVIDIA [training](https://www.olcf.ornl.gov/cuda-training-series/) with [exercises](https://github.com/olcf/cuda-training-series).