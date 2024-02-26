#include <iostream>
#include <cmath>
#include <cstdio>


double pi_approx(int n);
double f(int k);
double calc_error(double pi_app);


int main(){
    int n = 20;
    double computed_pi = pi_approx(n);
    std::printf("PI: %.16f \n", computed_pi);
    return 0;
}

double pi_approx(int n){
    double PI = 0.00, f_i = 0.00, pi_error = 0.00;
    for(int i = 0; i <= n; i++){
        std::cout.precision(16);
        std::cout.setf(std::ios::scientific);

        f_i = f(i);
        PI += f_i;
        pi_error = calc_error(PI);
        std::cout<<"n: "<<i<<"\t pi(n): "<<PI<<"\t error: "<<pi_error<<"\n";
    }
    return PI;
}

double f(int k){
    int y = 8*k;
    double 
        t = 4/(y + 1.00), 
        u = 2/(y + 4.00), 
        v = 1/(y + 5.00), 
        w = 1/(y + 6.00), 
        z = (1/std::pow(16,k));
        
    return (z*(t-u-v-w));
}

double calc_error(double pi_app){
    return std::abs(1 - (pi_app/M_PI));
}