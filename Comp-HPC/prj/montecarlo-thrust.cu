#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/async/copy.h>
#include <thrust/async/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <numeric>

int main(int argc, char **argv) {
  const double a = 0.0;
  const double b = 6.55;
  const int seed = 123456; 
  const int N = 22500*22500;
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<double> dist(a, b);
  thrust::host_vector<double> h_vec(N);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return std::exp(-1.0*std::pow(dist(rng), 2)); });

  //Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<double>());
  double y = (a-b)*x/N;
  printf("Sol: %.6f\n", y);
}