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

bool validate_params(const int PARAMS, const int argc)
{
  const bool USER_PARAMS = PARAMS + 1 == argc;
  if (!USER_PARAMS)
  {
    printf("ERROR: Insufficient number of parameters for performance.\n\tExpected %d parameters.\n\tGiven %d parameters.\n\n",
           PARAMS, argc - 1);
    return false;
  }
  else
  {
    return true;
  }
}

void rprintf(const char *fmt...)
{
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0')
  {
    if (*fmt == 's')
    {
      const char *s = va_arg(args, const char *);
      printf("%s:\t", s);
    }
    else if (*fmt == 'f')
    {
      double d = va_arg(args, double);
      printf("%.10f\n", d);
    }
    ++fmt;
  }

  va_end(args);
}

// functor to vector reduction
struct exp_square : public thrust::unary_function<double, double>
{
  __host__ __device__ double operator()(double x) const
  {
    return std::exp(-1.0 * x * x);
  }
};

int main(int argc, char **argv)
{
  const bool EXEC = validate_params(2, argc);
  if (!EXEC)
    return EXIT_FAILURE;
  const int N = std::atoi(argv[1]);
  const int seed = std::atoi(argv[2]);
  const double a = std::atof(argv[3]);
  const double b = std::atof(argv[4]);
  const float EXPECTED_VALUE = std::atof(argv[5]);

  auto t1 = std::chrono::high_resolution_clock::now();
  thrust::default_random_engine rng(seed);
  thrust::uniform_real_distribution<double> dist(a, b);
  thrust::host_vector<double> h_vec(N);
  thrust::generate(h_vec.begin(), h_vec.end(), [&]
                   { return dist(rng); });
  auto t2 = std::chrono::high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Time performing host_vector random number generation: " << ms_int.count() << "ms\n";

  // Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  typedef thrust::device_vector<double>::iterator DIterator;
  thrust::transform_iterator<exp_square, DIterator> iter(d_vec.begin(), exp_square());
  double x = thrust::reduce(iter, iter + N, 0.0, thrust::plus<double>());
  t2 = std::chrono::high_resolution_clock::now();
  ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  std::cout << "Time performing device_vector mapping and reduction: " << ms_int.count() << "ms\n";
  double y = (b - a) * x / N;
  printf("Sol: %.6f\n\n", y);

  return EXIT_SUCCESS;
}