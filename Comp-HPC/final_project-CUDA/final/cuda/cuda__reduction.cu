#include <cuda_runtime.h>
#include <curand.h>
#include "lib.h"

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

  init(&s0, (char **)argv, k0); // Initialize environment variables
  float *dev0, *dev1, *host;

  const size_t MEMORY_BYTES = s0.size * BYTES;
  const size_t SHARED_MEM = s0.threadsPerBlock * BYTES;
  const float EXPECTED_VALUE = std::atof(argv[6]);

  /* Fetch data from file created by integration execution */
  host = (float *)malloc(MEMORY_BYTES);
  auto t1 = std::chrono::high_resolution_clock::now();
  restoreData(host, argv[7]);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time restoring data from file: " << ms_int.count() << "us\n";

  /* Initialising memory */
  CUDA_CALL(cudaMalloc((void **)&dev0, MEMORY_BYTES));
  CUDA_CALL(cudaMalloc((void **)&dev1, MEMORY_BYTES));
  t1 = std::chrono::high_resolution_clock::now();
  CUDA_CALL(cudaMemcpy(dev0, host, MEMORY_BYTES, cudaMemcpyHostToDevice));
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time copying data from host to device: " << ms_int.count() << "us\n";

  // Launch the kernel
  t1 = std::chrono::high_resolution_clock::now();
  sumReduction<<<s0.blocksPerGrid, s0.threadsPerBlock, SHARED_MEM>>>(dev0, dev1, s0);
  sumReduction<<<1, s0.threadsPerBlock, SHARED_MEM>>>(dev1, dev1, s0);
  CUDA_CALL(cudaGetLastError());
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time performing device_vector reduction: " << ms_int.count() << "us\n";

  // Copy result back to host and compute output
  t1 = std::chrono::high_resolution_clock::now();
  CUDA_CALL(cudaMemcpy(host, dev1, MEMORY_BYTES, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaGetLastError());
  float x = ((s0.u) * (*host)) / (s0.size * s0.size);
  float err = fabs_err(x, EXPECTED_VALUE);
  rprintf("sfsf", "Result", x, "Error", err);
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "Time performing device_vector copy to host and compute result: " << ms_int.count() << "us\n";

  // Free memory
  free(host);
  CUDA_CALL(cudaFree(dev0));
  CUDA_CALL(cudaFree(dev1));

  return EXIT_SUCCESS;
}