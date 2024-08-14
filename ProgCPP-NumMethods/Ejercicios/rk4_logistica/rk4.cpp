#include "rk4.h"

//dN/dt=rN (1-N/K)
void f(const array &s, array &dsdt, double t){
  dsdt[0] = r*s[0]*(1.0-(s[0]/k));
}
void rk(int dim, double m, array &s, array &aux, array &k_i){
  for (int ii = 0; ii < dim; ++ii) 
    aux[ii] = s[ii] + m*k_i[ii];
}
void rk4(array &s, double t0, double tf, double h){
  int dim = s.size();
  array dsdt(dim), aux(dim), k1(dim), k2(dim), k3(dim), k4(dim);
  for(double t=t0; t <= tf; t+=h){

    f(s, k1, t);
    rk(dim, 0.5*h, s, aux, k1);
    //Computing k2
    f(aux, k2, t + 0.5*h);
    rk(dim, 0.5*h, s, aux, k2);
    //Computing k3
    f(aux, k3, t + 0.5*h);
    rk(dim, h, s, aux, k3);
    //Computing k4
    f(aux, k4, t + h);
    print(s, dsdt, t);
    for (int ii = 0; ii < s.size(); ++ii) {
      s[ii] = s[ii] + h*(k1[ii] + 2*k2[ii] + 2*k3[ii] + k4[ii])/6.0;
    }
  }
}
void print(const array &s, const array &dsdt, double t){
  std::cout << t << "\t" 
	    << s[0] << "\t" 
	    << dsdt[0] << "\n";
}



