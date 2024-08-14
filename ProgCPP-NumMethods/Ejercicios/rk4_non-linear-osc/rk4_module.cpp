/*
Make the oscillator non-linear, with a force of the form
F(x) = -kx^l for all l : l%2 = 0. 
How is the system behaviour as you change?
ProgCPP-NumMethods
RungeKuta 
*/
#include "rk4_module.h"
double force(double x){
    return -k*std::pow(x, lambda);
}
void f(const array &s, array &dsdt, double t, double FX){
  dsdt[0] = s[1];
  dsdt[1] = -w*w*s[0] + FX;
}
void rk(int dim, double m, array &s, array &aux, array &k_i){
  for (int ii = 0; ii < dim; ++ii) 
    aux[ii] = s[ii] + m*k_i[ii];
}
void rk4(array &s, double t0, double tf, double h){
  int dim = s.size();
  array dsdt(dim), aux(dim), k1(dim), k2(dim), k3(dim), k4(dim);
  for(double t=t0; t <= tf; t+=h){
    double fx = force(s[0]);
    //Computing k1
    f(s, k1, t, fx);
    rk(dim, 0.5*h, s, aux, k1);
    //Computing k2
    f(aux, k2, t + 0.5*h, fx);
    rk(dim, 0.5*h, s, aux, k2);
    //Computing k3
    f(aux, k3, t + 0.5*h, fx);
    rk(dim, h, s, aux, k3);
    //Computing k4
    f(aux, k4, t + h, fx);
    print(s, dsdt, t);
    for (int ii = 0; ii < s.size(); ++ii) {
      s[ii] = s[ii] + h*(k1[ii] + 2*k2[ii] + 2*k3[ii] + k4[ii])/6.0;
    }
  }
}
void print(const array &s, const array &dsdt, double t){
  std::cout << t << "\t" 
	    << s[0] << "\t" 
	    << s[1] << "\t" 
	    << dsdt[0] << "\t" 
	    << dsdt[1] << "\n";
}



