#include "vector_ops.hpp"
double mean(const std::vector<double> & data){
  double sum = 0.0;
  for(const auto i : data){
    sum += i;
  }
  return sum/(data.size()+0.0);
}
double re_error(double real, double computed){
    return std::fabs(real-computed)/std::fabs(real);
}