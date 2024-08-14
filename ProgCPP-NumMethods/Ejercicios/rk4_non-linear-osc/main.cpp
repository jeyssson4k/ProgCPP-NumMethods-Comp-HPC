#include "rk4_module.h"

int main(void){
    int dim = 2;
    //Initialize containers
    array s(dim), dsdt(dim);
    //Initial conditions
    s[0] = 1.0;
    s[1] = 1.27e-6;
    double t0 = 0.0, tf = 1024.55, h =0.5;
    
    //Performance rk4
    rk4(s, t0, tf, h);

    return EXIT_SUCCESS;
}