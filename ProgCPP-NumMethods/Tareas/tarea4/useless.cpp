#include "derivate.h"

int main(void){
    const std::vector<std::vector<double>> coefficients {
        { 0.0 }, 
        { -1.0, 1.0 }, 
        { -1.0/2.0, 1.0/2.0 },
        { 1.0/24.0, -27.0/24.0, 27.0/24.0, -1.0/24.0 },
        { 1.0/12.0, -2.0/3.0, 2.0/3.0, -1.0/12.0 },
        { -3.0/640.0, 25.0/384.0, -75.0/64.0, 75.0/64.0, -25.0/384.0, 3.0/640.0 }
    };

    const std::vector<std::vector<double>> h_scalars {
        { 0.0 },
        { -1.0/2.0, 1.0/2.0 },
        { -1.0, 1.0 },
        { -3.0/2.0, -1.0/2.0, 1.0/2.0, 3.0/2.0},
        { -2.0, -1.0, 1.0, 2.0 },
        { -5.0/2.0, -3.0/2.0, -1.0/2.0, 1.0/2.0, 3.0/2.0, 5.0/2.0 }
    };

    const std::vector<double> h {10.0, 1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16};
    const int N = h.size(); //samples
    const int M = 5; //orders
    const double x = M_PI;
    const double t = 2.0;

    const double theorical_df = 2*((x*std::cos(x)-std::sin(x))/(x*x));
    std::vector<double> df_extrapolation(M*N, 0.0);
    std::vector<double> df_central(M*N, 0.0);

    for(int k = 0; k < N; ++k){
        std::printf("%.16f\t", h[k]);
        for(int i = 1; i <= M; ++i){
            double df_0 = nm::r_extrapolation(nm::c_diff, [](double x){ return (2*std::sin(x))/x; }, i, x, h[k], t, coefficients[i], h_scalars[i]);
            double df_1 = nm::c_diff([](double x){ return 2*std::sin(x)/x; }, x, h[k], coefficients[i], h_scalars[i]);
            std::printf("%.16f\t %.16f\t", nm::df_error(theorical_df, df_0), nm::df_error(theorical_df, df_1));
            /*df_extrapolation[(i-1)*M+k] = nm::df_error(theorical_df, df_0);
            df_central[(i-1)*M+k] = nm::df_error(theorical_df, df_1);*/
        }
        std::printf("\n");
    }
    /*
    for(int j = 0; j < N; ++j){
        std::printf("%.17f\t", h[j]);
        for(int i = 0; i < M; ++i){
            std::printf("%.17f\t", df_central[i*M+j]);
        }
        for(int i = 0; i < M; ++i){
            std::printf("%.17f\t", df_extrapolation[i*M+j]);
        }
        std::printf("\n");
    }
    /*
    std::printf("\n\n*********DERIVADA CENTRAL**********\n");
    for(int i = 0; i < M; ++i){
        std::printf("Derivada de orden %d\n", (i+1));
        for(int j = 0; j < N; ++j){
            std::printf("%.17f\t",df_central[i*M+j]);
        }
        std::printf("\n");
    }
    std::printf("\n\n\n*******EXTRAPOLACION DE RICHARDSON*******\n");
    for(int i = 0; i < M; ++i){
        std::printf("Derivada de orden %d\n", (i+1));
        for(int j = 0; j < N; ++j){
            std::printf("%.17f\t",df_extrapolation[i*M+j]);
        }
        std::printf("\n");
    }
    */
    return EXIT_SUCCESS;
}