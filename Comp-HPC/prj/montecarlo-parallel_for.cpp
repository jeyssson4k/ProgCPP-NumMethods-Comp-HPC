#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <stdio.h>
#include <random>

double f(double x);

int main(int argc, char *argv[]){
    const int PARAMS = 3;
    const bool USER_PARAMS = PARAMS+1 == argc;
    if(!USER_PARAMS){
        std::printf("ERROR: Insufficient number of parameters for performance. \n\tExpected %d parameters. \n\tGiven %d parameters.\n\n", 
            PARAMS, argc-1);
        return EXIT_FAILURE;
    }

    const size_t N = static_cast<size_t>(std::atoll(argv[1]));
    const double L_LIMIT = std::atof(argv[2]);
    const double U_LIMIT = std::atof(argv[3]);
    const double EXPECTED_VALUE = 0.886227;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist(L_LIMIT, U_LIMIT);
    
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0ULL; i < N; i++){
        double r = dist(rng);
        sum += f(r);
    }

    double integrate = (U_LIMIT-L_LIMIT)*sum/N;
    std::printf("Integrate result: %.10f\n", integrate);
    double u = std::fabs(1 - (integrate/EXPECTED_VALUE));
    std::printf("Error: %.10f\n", u);
    return EXIT_SUCCESS;
}

double f(double x){
    return std::exp(-1.0*x*x);
}