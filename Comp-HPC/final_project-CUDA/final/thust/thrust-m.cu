#include <thrust/iterator/transform_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <stdio.h>
#include <cstdlib>
#include <cstdarg>
#include <chrono>

// note: functor inherits from unary_function
struct exp_square : public thrust::unary_function<double,double>
{
  __host__ __device__
  double operator()(double x) const
  {
    return std::exp(-1.0*x*x);
  }
};

int main(int argc, char **argv) {
  const double a = 0.0;
  const double b = 6.55;
  const int seed = 1234; 
  const int N = 6500*6500;

  auto t1 = std::chrono::high_resolution_clock::now();
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<double> dist(a, b);
  thrust::host_vector<double> h_vec(N);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });
  auto t2 = std::chrono::high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Time performing vector host random number generator: " << ms_int.count() << "ms\n";
  
  t1 = std::chrono::high_resolution_clock::now();
  //Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  typedef thrust::device_vector<double>::iterator DIterator;
  thrust::transform_iterator<exp_square, DIterator> iter(d_vec.begin(), exp_square());
  double x = thrust::reduce(iter, iter+N, 0.0, thrust::plus<double>());
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Time performing device_vector mapping and reduction: " << ms_int.count() << "ms\n";
  double y = (b-a)*x/N;
  printf("Sol: %.6f\n\n", y);

  return EXIT_SUCCESS;
}