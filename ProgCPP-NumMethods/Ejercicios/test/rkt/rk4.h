#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#define a 20.0
#define tau 5.0
#define k 0.12
#define file_output "r.txt"
typedef std::vector<double> array;

void f(const array &s, array &dsdt, double t);
void rk(int dim, double m, array &s, array &aux, array &k_i);
void rk4(array &s, double t0, double tf, double h);
void print(std::ofstream &out, const array &s, const array &dsdt, double t);
