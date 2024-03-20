#include "derivate.h"
#include <cstdio>
#include <fstream>
#include <iomanip>

int main(void){
    //Every w_i : weight given for each derivate
    const std::vector<std::vector<double>> coefficients {
        { 0.0 }, 
        { -1.0, 1.0 }, 
        { -1.0/2.0, 1.0/2.0 },
        { 1.0/24.0, -27.0/24.0, 27.0/24.0, -1.0/24.0 },
        { 1.0/12.0, -2.0/3.0, 2.0/3.0, -1.0/12.0 },
        { -3.0/640.0, 25.0/384.0, -75.0/64.0, 75.0/64.0, -25.0/384.0, 3.0/640.0 }
    };
    //Every c_i : factor to multiply by h 
    const std::vector<std::vector<double>> h_scalars {
        { 0.0 },
        { -1.0/2.0, 1.0/2.0 },
        { -1.0, 1.0 },
        { -3.0/2.0, -1.0/2.0, 1.0/2.0, 3.0/2.0},
        { -2.0, -1.0, 1.0, 2.0 },
        { -5.0/2.0, -3.0/2.0, -1.0/2.0, 1.0/2.0, 3.0/2.0, 5.0/2.0 }
    };
    //every error order
    const std::vector<int> orders {0,2,2,4,4,6};

    //All samples for h
    const std::vector<double> h {10.0, 1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16};
    //Specific value to evaluate
    const double x = M_PI;
    //t factor for richardson extrapolation
    const double t = 2.0;

    //Expected value for f'(x)
    const double theorical_df = 2*((x*std::cos(x)-std::sin(x))/(x*x));

    //creates an instance of fstream to write in a file
    const std::string output_data_file_name = "derivates.txt";
    std::ofstream outputFile(output_data_file_name);
    outputFile << std::fixed << std::setprecision(20);

    for(int k = 0; k < h.size(); ++k){
        //std::printf("%.16f\t", h[k]);
        outputFile << h[k] << "\t";
        for(int i = 1; i <= 5; ++i){
            //calculates the richardson extrapolation, f(x) is a lambda function, i is order, 
            double df_0 = nm::r_extrapolation(nm::c_diff, [](double x){ return (2*std::sin(x))/x; }, orders[i], x, h[k], t, coefficients[i], h_scalars[i]);
            double df_1 = nm::c_diff([](double x){ return 2*std::sin(x)/x; }, x, h[k], coefficients[i], h_scalars[i]);
            outputFile 
                    << nm::df_error(theorical_df, df_0) << "\t" 
                    << nm::df_error(theorical_df, df_1) << "\t";
            
            //std::printf("%.16f\t %.16f\t", nm::df_error(theorical_df, df_0), nm::df_error(theorical_df, df_1));
        }
        outputFile << "\n";
        //std::printf("\n");
    }
    outputFile.close();
    return EXIT_SUCCESS;
}