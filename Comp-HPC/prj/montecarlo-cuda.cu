#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize> //unroll loop
__global__ void cuda_integrate(float *data, double *resp, size_t size, double a, double b){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y*blockDim.y + threadIdx.y;

    if(col == 0 && row < size){
        double sum = 0.0f;
        for(int i = 0; i < size; ++i){
            double x = (b-a)*data[row*size+i];
            sum += std::exp(-1.0*x*x);
        }
        resp[row] = sum;
    }

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += resp[i] + resp[i+blockSize]; i += gridSize; }
    __syncthreads(); 

    //512 is the limit 
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) resp[blockIdx.x] = (b-a)*sdata[0]/size;
    
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
    const int PARAMS = 5;
    const bool USER_PARAMS = PARAMS+1 == argc;
    if(!USER_PARAMS){
        printf(
            "ERROR: Insufficient number of parameters for performance.
            \n\tExpected %d parameters.
            \n\tGiven %d parameters.\n\n", 
            PARAMS, argc-1);
        return EXIT_FAILURE;
    }

    const size_t M = static_cast<size_t>(std::atoll(argv[1]));
    const size_t seed = static_cast<size_t>(std::atoll(argv[2]));
    const size_t th = static_cast<size_t>(std::atoll(argv[3]));
    const double a = std::atof(argv[4]);
    const double b = std::atof(argv[5]);

    const size_t size = M*M;

    float  *devData;
    double *hostData, *devResp;

    //grid shape = general division n*m
    //block shape = division for each position in grid p*q threads
    dim3 block_shape = dim3(th,th);
    dim3 grid_shape = dim3(
        max(1.0, ceil((float) M / (float)block_shape.x)),
        max(1.0, ceil((float) M / (float)block_shape.y))
    );

    /* Allocate doubles on host */
    hostData = (double *)calloc(M, sizeof(double));
    
    /* Allocate doubles on device */
    CUDA_CALL(cudaMalloc((void **)&devData, size*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&devResp, M*sizeof(double)));

    curandGenerator_t gen;
        
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    /* Generate size doubles on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, size));

    /* Exec Montecarlo method */
    cuda_integrate<<<grid_shape, block_shape>>>(devData, devResp, M, a, b);
    cudaDeviceSynchronize();

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devResp, M*sizeof(double),cudaMemcpyDeviceToHost));
    printf("Result:\t%.6f\t", *hostData);
    
    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    CUDA_CALL(cudaFree(devResp));
    free(hostData);

    return EXIT_SUCCESS;
}