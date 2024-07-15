#include <cuda_runtime.h>
#include <curand.h>
#include <nvfunctional>
#include "lib.h"

__device__ float f(float x) { return std::exp(-1.0 * x * x); }
__global__ void sumReduction(float *v, float *v_r, bool isInitialLoading, S st)
{
  /* Initialize components */
  extern __shared__ float partial_sum[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  nvstd::function<float(float)> fn = f;

  // Load elements into shared memory
  if (isInitialLoading)
  {
    float y = fn((st.b - st.a) * v[tid]);
    partial_sum[threadIdx.x] = y;
    __syncthreads();
  }
  else
  {
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();
  }
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = 1; s < blockDim.x; s <<= 1)
  {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x)
    {
      partial_sum[index] += partial_sum[index + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0)
  {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

int main(int argc, char **argv)
{
  const bool EXEC = validate_params(7, argc);
  if (!EXEC)
    return EXIT_FAILURE;
  enum K k0 = CUDA_REDUCTION; // Define type of execution
  S s0{};

  /* Initialising memory */
  init(&s0, (char **)argv, k0);
  float *dev0, *dev1, *host;
  const size_t MEMORY_BYTES = s0.size * BYTES;
  const size_t SHARED_MEM = s0.threadsPerBlock * BYTES;
  const float EXPECTED_VALUE = std::atof(argv[6]);
  CUDA_CALL(cudaMalloc((void **)&dev0, MEMORY_BYTES));
  CUDA_CALL(cudaMalloc((void **)&dev1, MEMORY_BYTES));

  auto t1 = std::chrono::high_resolution_clock::now();
  /* Declare generator */
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, s0.seed));
  /* Generate n floats on device */
  CURAND_CALL(curandGenerateUniform(gen, dev0, s0.size));
  CUDA_CALL(cudaGetLastError());
  auto t2 = std::chrono::high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time performing device_vector random numbers generation: " << ms_int.count() << "us\n";

  // Launch the kernel
  t1 = std::chrono::high_resolution_clock::now();
  sumReduction<<<s0.blocksPerGrid, s0.threadsPerBlock, SHARED_MEM>>>(dev0, dev1, true, s0);
  sumReduction<<<1, s0.threadsPerBlock, SHARED_MEM>>>(dev1, dev1, false, s0);
  // CUDA_CALL(cudaGetLastError());
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time performing device_vector mapping and reduction: " << ms_int.count() << "us\n";

  // Copy result back to host and compute output
  t1 = std::chrono::high_resolution_clock::now();
  host = (float *)malloc(MEMORY_BYTES);
  CUDA_CALL(cudaMemcpy(host, dev1, MEMORY_BYTES, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());
  float x = ((s0.u) * (*host)) / (s0.size);
  float err = fabs_err(x, EXPECTED_VALUE);
  rprintf("sfsf", "Result", x, "Error", err);
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time performing device_vector copy to host and compute result: " << ms_int.count() << "us\n";

  // Free memory
  free(host);
  CUDA_CALL(cudaFree(dev0));
  CUDA_CALL(cudaFree(dev1));
  CURAND_CALL(curandDestroyGenerator(gen));

  return EXIT_SUCCESS;
}