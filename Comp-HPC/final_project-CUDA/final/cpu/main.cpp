#include "utils.h"

int main(int argc, char *argv[]){
    const bool EXEC = utils::validate_params(5, argc);
    if(!EXEC) return EXIT_FAILURE;
    
    //Get data from command line
    const size_t N = static_cast<size_t>(std::atoll(argv[1]));
    const double L_LIMIT = std::atof(argv[2]);
    const double U_LIMIT = std::atof(argv[3]);
    const double EXPECTED_VALUE = std::atof(argv[4]);
    const bool IS_PARALLEL = static_cast<bool>(std::atoi(argv[5]));

    printf("Executing on CPU\n"); 
    utils::MontecarloIO integrate = utils::MontecarloIO(L_LIMIT, U_LIMIT, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    if(IS_PARALLEL){
        integrate.montecarlo_OMP([](double x){return std::exp(-1.0*x*x);});
    }else{
        integrate.montecarlo([](double x){return std::exp(-1.0*x*x);});
    }

    integrate.computeResult();
    integrate.computeError(EXPECTED_VALUE);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Time performing integral: " << ms_int.count() << "ms\n";
    double u = integrate.getRes();
    double v = integrate.getError();
    utils::rprintf("sfsf","Result", u, "Error", v);
    
    return EXIT_SUCCESS;
}