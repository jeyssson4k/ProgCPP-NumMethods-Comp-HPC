// stats.cpp
#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <numeric>

void fill(std::vector<double> & array);
void stats(const std::vector<double> & array, double &mean, double &sigma);

int main(int argc, char *argv[])
{
  const int N = std::atoi(argv[1]);
  std::vector<double> data(N);

  // llenar el arreglo
  fill(data);

  // calcular stats
  double mean{0.0}, sigma{0.0};
  double start = omp_get_wtime();
  stats(data, mean, sigma);
  double time = omp_get_wtime() - start;
  std::printf("%.15le\t\t%.15le\t\t%.15le\n", mean, sigma, time);

  return 0;
}

void fill(std::vector<double> & array)
{
  const int N = array.size();
#pragma omp parallel for
  for(int ii = 0; ii < N; ii++) {
      array[ii] = 2*ii*std::sin(std::sqrt(ii/56.7)) +
        std::cos(std::pow(1.0*ii*ii/N, 0.3));
  }
}

void stats(const std::vector<double> & array, double & mean, double & sigma)
{
  int N = array.size();
  double suma = 0.0;
#pragma omp parallel for reduction(+:suma)
  for(int ii = 0; ii < N; ii++) {
    suma += array[ii];
  }
  mean = suma/N;
}