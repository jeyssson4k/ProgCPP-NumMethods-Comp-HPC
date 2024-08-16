#include "rk4.h"

//dN/dt=rN (1-N/K)
void f(const array &s, array &dsdt, double t){
  dsdt[0] = (1./tau)*(a-s[0])-k*s[0];
  dsdt[1] = (1./tau)*(s[0]-s[1])-k*s[1];
  dsdt[2] = (-1./tau)*(s[2])+k*s[0];
  dsdt[3] = (1./tau)*(s[2]-s[3])+k*s[1];
}
void rk(int dim, double m, array &s, array &aux, array &k_i){
  for (int ii = 0; ii < dim; ++ii) 
    aux[ii] = s[ii] + m*k_i[ii];
}
void rk4(array &s, double t0, double tf, double h){
  int dim = s.size();
  array dsdt(dim), aux(dim), k1(dim), k2(dim), k3(dim), k4(dim);
  std::ofstream output(file_output);
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
    print(output, s, dsdt, t);
    for (int ii = 0; ii < s.size(); ++ii) {
      s[ii] = s[ii] + h*(k1[ii] + 2*k2[ii] + 2*k3[ii] + k4[ii])/6.0;
    }
  }
}
void print(std::ofstream &out, const array &s, const array &dsdt, double t){
  out 
    << t << "\t"
    << s[0] << "\t"
    << s[1] << "\t"
    << s[2] << "\t"
    << s[3] << "\n";
}



