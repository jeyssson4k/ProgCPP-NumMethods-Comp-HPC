#include <cuda_runtime.h>
#include <curand.h>
#include "lib.h"

__device__ float f(float x) { return std::exp(-1.0 * x * x); }

__global__ void sumReduction(float *v, float *v_r, S st)
{
  /* Initialize components */
  extern __shared__ float partial_sum[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    // Each thread does work unless it is further than the stride
    if (threadIdx.x < s)
    {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is indexed by this block
  if (threadIdx.x == 0)
    v_r[blockIdx.x] = partial_sum[0];
}

int main(int argc, char **argv)
{
  /* Verify if all of params exist */
  const bool EXEC = validate_params(7, argc);
  if (!EXEC)
    return EXIT_FAILURE;

  enum K k0 = CUDA_REDUCTION; // Define type of execution
  S s0{};
  H h0{};
  D d0{};
  init(&s0, (char **)argv, k0); // Initialize environment variables

  const size_t MEMORY_BYTES = s0.size * BYTES;
  const size_t SHARED_MEM = s0.threadsPerBlock * BYTES;
  const float EXPECTED_VALUE = std::atof(argv[6]);

  /* Fetch data from file created by integration execution */
  h0.h_o = (float *)malloc(MEMORY_BYTES);
  clock_t tStart = clock();
  restoreData(h0.h_o, argv[7]);
  printf("Time taken: %.3fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  /* Initialising memory */
  CUDA_CALL(cudaMalloc((void **)&d0.d_i, MEMORY_BYTES));
  CUDA_CALL(cudaMalloc((void **)&d0.d_o, MEMORY_BYTES));
  CUDA_CALL(cudaMemcpy(d0.d_i, h0.h_o, MEMORY_BYTES, cudaMemcpyHostToDevice));

  // Launch the kernel
  sumReduction<<<s0.blocksPerGrid, s0.threadsPerBlock, SHARED_MEM>>>(d0.d_i, d0.d_o, s0);
  sumReduction<<<1, s0.threadsPerBlock, SHARED_MEM>>>(d0.d_o, d0.d_o, s0);
  CUDA_CALL(cudaGetLastError());

  // Copy result back to host and compute output
  CUDA_CALL(cudaMemcpy(h0.h_o, d0.d_o, MEMORY_BYTES, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());
  float x = ((s0.b - s0.a) * (*h0.h_o)) / (s0.size * s0.size);
  float err = fabs_err(x, EXPECTED_VALUE);
  rprintf("sfsf", "Result", x, "Error", err);

  // Free memory
  free(h0.h_o);
  CUDA_CALL(cudaFree(d0.d_i));
  CUDA_CALL(cudaFree(d0.d_o));

  return EXIT_SUCCESS;
}