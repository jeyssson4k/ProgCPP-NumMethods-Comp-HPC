#pragma once
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdarg>
#include <tuple>
#include <limits>

namespace linreg{
    template<typename T>
    double map(T f_x, const std::vector<double> &v){
        double s = 0.;
        if(v.size() < 1) return s;

        for(const double v_i : v){
            s += f_x(v_i);
        }
        return s;
    }
    template<typename T>
    double dual_map(T f_xy, const std::vector<double> &u, const std::vector<double> &v){
        double s = 0.;
        if(u.size() != v.size()){
            std::cerr << 
                "--------------------------------------------------------------------------------\n" << 
                "Argument error [in dual_map]: expected u and v with the same size." <<
                "\n\t u.size: " << u.size() <<
                "\n\t v.size: " << v.size() <<
                "\n\t Function will return: " << s << "\n" <<
                "--------------------------------------------------------------------------------\n";
            return s;
        }else{
            for(int i = 0; i < u.size(); ++i){
                s += f_xy(u[i], v[i]);
            }
            return s;
        }
        
    }

    std::vector<std::vector<double>> fetch(const std::string &path);
    void show(const std::vector<double> summary);
    void show_error(const double x, const double y, std::string var);
    double r_error(const double x, const double y);
    double vector_max(const std::vector<double> &v);
    double vector_min(const std::vector<double> &v);
    std::vector<double> linspace(double a, double b, int steps);
    std::tuple<double, double, double, double, double> linregress(const std::vector<double>& x, const std::vector<double>& y, bool show_results);
    std::vector<double> expected_x(const std::vector<double> &x, int steps);
    std::vector<double> expected_y(const std::vector<double> &x, double slope, double intercept);
}

