#include "lib.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <nvfunctional>

__device__ float f(float x) { return std::exp(-1.0 * x * x); }
__global__ void cuda_integrate(float *data, float *resp, S s)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    nvstd::function<float(float)> fn = f;

    if (col == 0 && row < s.size)
    {
        float sum = 0.0f;
        for (int i = 0; i < s.size; i += 16)
        {
            int t = row * s.size + i;
            float x_0 = (data[t] != NULL) ? s.u * data[t] : 0.0f,
                  x_1 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_2 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_3 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_4 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_5 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_6 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_7 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_8 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_9 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_10 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_11 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_12 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_13 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_14 = (data[++t] != NULL) ? s.u * data[t] : 0.0f,
                  x_15 = (data[++t] != NULL) ? s.u * data[t] : 0.0f;
            sum += (fn(x_0) + fn(x_1) + fn(x_2) + fn(x_3) + fn(x_4) +
                    fn(x_5) + fn(x_6) + fn(x_7) + fn(x_8) + fn(x_9) + fn(x_10) +
                    fn(x_11) + fn(x_12) + fn(x_13) + fn(x_14) + fn(x_15));
        }
        resp[row] = sum;
    }
}

int main(int argc, char *argv[])
{
    /* Verify if all of params exist */
    const bool EXEC = validate_params(7, argc);
    if (!EXEC)
        return EXIT_FAILURE;

    enum K k0 = CUDA_INTEGRATION; // Define type of execution
    S s0{};
    init(&s0, (char **)argv, k0);    // Initialize environment variables
    float *devData, *devResp, *host; // Float Pinters to save the data

    const size_t SIZE = s0.size * s0.size;                           // Matrix data size
    const size_t MATRIX_BYTES = SIZE * BYTES;                        // Matrix data: memory allocation
    const size_t MEMORY_BYTES = s0.size * BYTES;                     // Response data vector: memory allocation
    dim3 block_shape = dim3(s0.threadsPerBlock, s0.threadsPerBlock); // Threads per block
    dim3 grid_shape = dim3(s0.blocksPerGrid, s0.blocksPerGrid);      // Blocks per Grid

    printf("Executing on GPU using %zu x %zu data\n\n", s0.size, s0.size);
    printf("Block Shape: %d x %d\n", s0.threadsPerBlock, s0.threadsPerBlock);
    printf("Grid Shape: %zu x %zu\n", s0.blocksPerGrid, s0.blocksPerGrid);

    /* Allocate floats */
    CUDA_CALL(cudaMalloc((void **)&devData, MATRIX_BYTES));
    CUDA_CALL(cudaMalloc((void **)&devResp, MEMORY_BYTES));
    host = (float *)malloc(MEMORY_BYTES);

    /* Create pseudo-random number generator */
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, s0.seed)); // Set the seed
    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, SIZE));

    // Compute values of function e^(-x^2) with random numbers as input
    cuda_integrate<<<grid_shape, block_shape>>>(devData, devResp, s0);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpy(host, devResp, MEMORY_BYTES, cudaMemcpyDeviceToHost));

    // Write vector result in a file with CPU
    clock_t tStart = clock();
    const char *data_file = argv[7];
    std::ofstream outputFile(data_file);
    outputFile << std::fixed << std::setprecision(20);

#pragma omp parallel for
    for (int i = 0; i < s0.size; ++i)
    {
        outputFile << host[i] << "\n";
    }
    outputFile.close();
    printf("Time taken: %.3fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    CUDA_CALL(cudaFree(devResp));
    free(host);

    return EXIT_SUCCESS;
}