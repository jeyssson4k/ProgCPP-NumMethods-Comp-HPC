#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#define r 0.674002
#define k 23.987
typedef std::vector<double> array;

void f(const array &s, array &dsdt, double t);
void rk(int dim, double m, array &s, array &aux, array &k_i);
void rk4(array &s, double t0, double tf, double h);
void print(const array &s, const array &dsdt, double t);
