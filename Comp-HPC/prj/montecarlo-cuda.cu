#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#define CUDA_CALL(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return EXIT_FAILURE; \
    } \
} while (0)

__global__ void cuda_integrate(float *data, float *resp, int size, float a, float b){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if(col == 0 && row < size){
        float sum = 0.0f;
        for(int i = 0; i < size; ++i){
            float x = (b-a)*data[row*size+i];
            sum += std::exp(-1.0*x*x);
        }
        resp[row] = sum;
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
    const int PARAMS = 5;
    if(!(PARAMS+1 == argc)){
        printf("ERROR: Insufficient number of parameters for performance. \n\tExpected %d parameters. \n\tGiven %d parameters.\n\n", 
            PARAMS, argc-1);
        return EXIT_FAILURE;
    }

    //line command inputs
    const size_t M = static_cast<size_t>(std::atoll(argv[1]));
    const size_t seed = static_cast<size_t>(std::atoll(argv[2]));
    const size_t th = static_cast<size_t>(std::atoll(argv[3]));
    const float a = std::atof(argv[4]);
    const float b = std::atof(argv[5]);

    printf("Executing on GPU using %zu x %zu data\n\n", M, M);
    //matrix size for random generator
    const size_t size = M*M;
    //device pointers to save data
    float *devData, *devResp;

    //grid shape = general division n*m
    //block shape = division for each position in grid p*q threads
    dim3 block_shape = dim3(th,th);
    printf("Block Shape: %zu x %zu\n", th, th);

    /*
    float gshx = max(1.0, ceil((float) M/(float)block_shape.x));
    float gshy = max(1.0, ceil((float) M/(float)block_shape.y));
    dim3 grid_shape = dim3(gshx, gshy);

    printf("Grid Shape: %.2f x %.2f\n", gshx, gshy);
    */
    
    dim3 grid_shape = dim3(1024, 1024);
    printf("Block Shape: 1024 x 1024\n");

    /* Allocate floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, size*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&devResp, M*sizeof(float)));

    curandGenerator_t gen;  
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, size));

    //compute values of function e^(-x^2) with random inputs
    cuda_integrate<<<grid_shape, block_shape>>>(devData, devResp, M, 0.0f,6.55f);
    cudaDeviceSynchronize();

    //computing results
    thrust::device_vector<float> dv{devResp, devResp + (M)};
    float y = thrust::reduce(dv.begin(), dv.end(), 0.0, thrust::plus<float>());
    float r = (b-a)*y/size;
    printf("Result: %.10f\t", r);
    double u = std::fabs(1 - (r/0.886227));
    std::printf("Error: %.10f\n", u);
    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    CUDA_CALL(cudaFree(devResp));

    return EXIT_SUCCESS;
}