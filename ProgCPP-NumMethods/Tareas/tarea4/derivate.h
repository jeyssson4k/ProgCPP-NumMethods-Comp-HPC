#pragma once
#include <cmath>
#include <vector>
#include <cstdio>

using fptr = double(double);
using dfptr = double(fptr, double, double, std::vector<double>, std::vector<double>);
namespace nm {
    double r_extrapolation(dfptr df, fptr f, int alpha, double x, double h, double t, std::vector<double> c, std::vector<double> scalars);
    double c_diff(fptr f, double x, double h, std::vector<double> c, std::vector<double> scalars);
    double df_error(double dy_theo, double dy_num);
}