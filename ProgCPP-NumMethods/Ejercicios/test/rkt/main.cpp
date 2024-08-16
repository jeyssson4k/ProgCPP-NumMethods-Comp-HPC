#include "rk4.h"

int main(void){
    //Initialize containers
    array s{0.0, 0.0, 0.0, 0.0};
    double t0 = 0.0, tf = 50.0, h = 0.01;
    
    //Performance rk4
    rk4(s, t0, tf, h);

    return EXIT_SUCCESS;
}