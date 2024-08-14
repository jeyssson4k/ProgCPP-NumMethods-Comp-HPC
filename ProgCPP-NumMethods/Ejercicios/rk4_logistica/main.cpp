//dN/dt=rN (1-N/K)
#include "rk4.h"

int main(void){
    int dim = 1;
    //Initialize containers
    array s(dim), dsdt(dim);
    //Initial conditions
    s[0] = 1.27e-6;

    double t0 = 0.0, tf = 48.65, h = 0.5;
    
    //Performance rk4
    rk4(s, t0, tf, h);

    return EXIT_SUCCESS;
}