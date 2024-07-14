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
  for (int s = 1; s < blockDim.x; s *= 2)
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
  H h0{};
  D d0{};

  /* Initialising memory */
  init(&s0, (char **)argv, k0);
  const size_t MEMORY_BYTES = s0.size * BYTES;
  const size_t SHARED_MEM = s0.threadsPerBlock * BYTES;
  const float EXPECTED_VALUE = std::atof(argv[6]);
  CUDA_CALL(cudaMalloc((void **)&d0.d_i, MEMORY_BYTES));
  CUDA_CALL(cudaMalloc((void **)&d0.d_o, MEMORY_BYTES));

  /* Declare generator */
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, s0.seed));
  /* Generate n floats on device */
  CURAND_CALL(curandGenerateUniform(gen, d0.d_i, s0.size));
  // Launch the kernel
  sumReduction<<<s0.blocksPerGrid, s0.threadsPerBlock, SHARED_MEM>>>(d0.d_i, d0.d_o, true, s0);
  sumReduction<<<1, s0.threadsPerBlock, SHARED_MEM>>>(d0.d_o, d0.d_o, false, s0);
  // CUDA_CALL(cudaGetLastError());

  // Copy result back to host and compute output
  h0.h_o = (float *)malloc(MEMORY_BYTES);
  CUDA_CALL(cudaMemcpy(h0.h_o, d0.d_o, MEMORY_BYTES, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());
  float x = ((s0.b - s0.a) * (*h0.h_o)) / (s0.size);
  float err = fabs_err(x, EXPECTED_VALUE);
  rprintf("sfsf", "Result", x, "Error", err);

  // Free memory
  free(h0.h_o);
  CUDA_CALL(cudaFree(d0.d_i));
  CUDA_CALL(cudaFree(d0.d_o));

  return EXIT_SUCCESS;
}