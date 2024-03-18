#include "derivate.h"

double nm::r_extrapolation(dfptr df, fptr f, int alpha, double x, double h, double t, std::vector<double> c, std::vector<double> scalars){
    double t_exp_alpha = std::pow(t, alpha);
    double df_0 = df(f, x, h/t, c, scalars);
    double df_1 = df(f, x, h, c, scalars);

    return ((t_exp_alpha*df_0) - df_1)/(t_exp_alpha-1);
}

double nm::c_diff(fptr f, double x, double h, std::vector<double> c, std::vector<double> scalars){
    double df = 0.0;
    for(int i = 0; i < scalars.size(); ++i){
        double x_0 = x+(scalars[i]*h);
        df += c[i]*f(x_0);
    }
    return df/h;
}

double nm::df_error(double dy_theo, double dy_num){
    return std::fabs(dy_theo-dy_num)/std::fabs(dy_theo);
}