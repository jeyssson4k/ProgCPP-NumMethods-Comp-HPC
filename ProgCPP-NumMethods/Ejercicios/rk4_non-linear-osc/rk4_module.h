#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#define w 1.2345
#define k 0.748753
#define b 0.284433
#define lambda 2.0
typedef std::vector<double> array;

double force(double x);
double force(double m, double v);
void f(const array &s, array &dsdt, double t);
void rk(int dim, double m, array &s, array &aux, array &k_i);
void rk4(array &s, double t0, double tf, double h);
void print(const array &s, const array &dsdt, double t);
