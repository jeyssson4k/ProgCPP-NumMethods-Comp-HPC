#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

__global__ void cuda_integrate(float *data, double *resp, int size, double a, double b){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if(col == 0 && row < size){
        double sum = 0.0f;
        for(int i = 0; i < size; ++i){
            double x = b*data[row*size+i] - a*data[row*size+i];
            //printf("%.6f\n", x);
            sum += std::exp(-1.0*x*x);
        }
        resp[row] = sum;
        //printf("%.6f\n", sum);
    }
    
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
    const int PARAMS = 3;
    const bool USER_PARAMS = PARAMS+1 == argc;
    if(!USER_PARAMS){
        printf("ERROR: Insufficient number of parameters for performance. \n\tExpected %d parameters. \n\tGiven %d parameters.\n\n", 
            PARAMS, argc-1);
        return EXIT_FAILURE;
    }
    size_t M = static_cast<size_t>(std::atoll(argv[1]));
    size_t seed = static_cast<size_t>(std::atoll(argv[2]));
    size_t th = static_cast<size_t>(std::atoll(argv[3]));
    size_t size = M*M;

    float *devData;
    double *hostData, *devResp;

    //grid shape = general division n*m
    //block shape = division for each position in grid p*q threads
    dim3 block_shape = dim3(th,th);
    dim3 grid_shape = dim3(
        max(1.0, ceil((double) M/(double)block_shape.x)),
        max(1.0, ceil((double) M/(double)block_shape.y))
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
    /* Generate n doubles on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, size));

    
    cuda_integrate<<<grid_shape, block_shape>>>(devData, devResp, M, 0.0f,6.55f);
    cudaDeviceSynchronize();

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devResp, M*sizeof(double),cudaMemcpyDeviceToHost));
    
    /*
    double h = 0.0f;
    for(size_t i = 0; i < M; ++i){
        h += hostData[i];
    }
    */
    thrust::device_vector<double> dv{hostData, hostData+(M)};
    double y = thrust::reduce(dv.begin(), dv.end(), 0.0, thrust::plus<double>());
    double integrate_res = (6.55-0.0)*y/size;
    printf("Result:\t%.6f\t", integrate_res);
    
    /* Allocate doubles on host 
    hostData = (double *)calloc(M*M, sizeof(double));
    */

    /* Copy device memory to host 
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(double),
        cudaMemcpyDeviceToHost));
    */
    
    //cudaDeviceSynchronize();

    /* Show result 
    for(size_t i = 0ULL; i < n; i++) {
        printf("%llu\t%1.4f\n", i, hostData[i]);
    }
    */
    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    CUDA_CALL(cudaFree(devResp));
    free(hostData);
    return EXIT_SUCCESS;
}