/*
dy/dt = v
d2y/dt2 = (1-y²)v - y 
*/
#include "rk4.h"

int main(void){
    int dim = 2;
    //Initialize containers
    array s(dim), dsdt(dim);
    //Initial conditions
    s[0] = 1.0;
    s[1] = 1.0;

    double t0 = 0.0, tf = 10.0, h = 0.01;
    
    //Performance rk4
    rk4(s, t0, tf, h);

    return EXIT_SUCCESS;
}