#include "utils.hpp"

std::vector<std::vector<double>> linreg::fetch(const std::string &path){
    std::vector<std::vector<double>> data;
    std::vector<double> u, v;
    std::ifstream data_file(path);
    double x = 0., y = 0.;
    int it = 0;
    while(data_file >> x >> y){
        //std::printf("x: %.2f \t y: %.2f\n", x, y);
        u.push_back(x); v.push_back(y); it++;
    }
    if(it == 0){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Fetching error [in fetch]: file not found or file is empty!" <<
            "\n\t u.size: " << v.size() <<
            "\n\t v.size: " << v.size() <<
            "\n\t Try to use a valid path. \n" <<
            "--------------------------------------------------------------------------------\n";
    }
    data.push_back(u); data.push_back(v);
    data_file.close();
    return data;
}
void linreg::show(const std::vector<double> summary){
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf(
        "Summary\n\t slope: %.5e +- %.5e\n\t intercept: %.5e +- %.5e\n\t s: %.5e \n\n", 
        summary[0], summary[1], summary[2], summary[3], summary[4]
    );
    std::printf("Linear Regression executed successfully!\n");
    std::printf("--------------------------------------------------------------------------------\n");
}
void linreg::show_error(const double x, const double y, std::string var){
    const double err = linreg::r_error(x,y);
     const char* char_txt = var.c_str();
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf(
        "Error summary for %s\n\t Expected value: %.16e\n\t Current value: %.16e\n\t Error: %.16e\n",char_txt, y, x, err
    );
    std::printf("--------------------------------------------------------------------------------\n");    
}
double linreg::r_error(const double x, const double y){
    if(x == y) return 0.;
    if(std::fabs(y-std::numeric_limits<double>::min()) < 1e-300){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Math error [in r_error]: y is too near at the numerical limit." <<
            "\n\t y: " << y <<
            "\n\t limit: " << std::numeric_limits<double>::min() <<
            "\n\t Try to avoid a zero division error. \n" <<
            "--------------------------------------------------------------------------------\n";
        return -1.;
    }
    return std::fabs(1 - (x/y));
}
double linreg::vector_max(const std::vector<double> &v){
    if(v.size() == 0) return 0.;
    double max = v[0];
    for(const double v_i : v){
        if(v_i > max){
            max = v_i;
        }
    }
    return max;
}
double linreg::vector_min(const std::vector<double> &v){
    if(v.size() == 0) return 0.;
    double min = v[0];
    for(const double v_i : v){
        if(v_i < min){
            min = v_i;
        }
    }
    return min;
}
std::vector<double> linreg::linspace(double a, double b, int steps){
    std::vector<double> w; 
    if(b<a){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument error [in linspace]: Impossible to generate an array of evenly spaced numbers over a specified interval." <<
            "\n\t Probably the interval should be inverted."
            "\n\t a: " << a <<
            "\n\t b: " << b <<
            "\n\t Try to call linspace again with b > a. \n" <<
            "--------------------------------------------------------------------------------\n";
        return w;
    }else if(b == a){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument error [in linspace]: Impossible to generate an array of evenly spaced numbers over a specified interval." <<
            "\n\t There are no possible values between a and b."
            "\n\t a: " << a <<
            "\n\t b: " << b <<
            "\n\t Try to call linspace again with b > a. \n" <<
            "--------------------------------------------------------------------------------\n";
        return w;
    }
    const double step = (b-a)/steps;
    double x_i = a;
    while(x_i <= b){
        w.push_back(x_i);
        x_i+=step;
    }

    return w;
}
std::tuple<double, double, double, double, double> linreg::linregress(const std::vector<double>& x, const std::vector<double>& y, bool show_results){
    double N = 0., slope = 0., intercept = 0., s = 0., delta_slope = 0., delta_intercept = 0.;
    if(x.size()-y.size() == 0){
        N = static_cast<double>(x.size());
    }else{
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument error [in linear regression]: expected x and y with the same size." <<
            "\n\t x.size: " << x.size() <<
            "\n\t y.size: " << y.size() <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
        return std::make_tuple(slope, delta_slope, intercept, delta_intercept, s);
    }
    auto f = [](double x){return x;};
    auto g = [](double x){return x*x;};
    auto h = [](double x, double y){return x*y;};

    const double 
        k    = 1/N,
        e_x  = k*linreg::map(f, x),
        e_y  = k*linreg::map(f, y),
        e_xx = k*linreg::map(g, x),
        e_yy = k*linreg::map(g, y),
        e_xy = k*linreg::dual_map(h, x, y);
    
    if(linreg::r_error(e_xx, e_x) < 1e-4){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Unexpected error [in linear regression]: error processing data to build the linear regression." <<
            "\n\t Factors E_x, E_xx are too much near." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
        return std::make_tuple(slope, delta_slope, intercept, delta_intercept, s);
    }
    slope           = (e_xy - e_x*e_y)/(e_xx - (e_x*e_x));
    intercept       = ((e_xx*e_y) - (e_x*e_xy))/(e_xx - (e_x*e_x));
    s               = (N/(N-2))*(e_yy - (e_y*e_y) - (slope*slope)*(e_xx - (e_x*e_x)));
    delta_slope     = (s/(N*(e_xx - (e_x*e_x))));
    delta_intercept = (delta_slope*e_xx);

    if(show_results){
        linreg::show({slope, delta_slope, intercept, delta_intercept, s});
    }

    return std::make_tuple(slope, std::sqrt(delta_slope), intercept, std::sqrt(delta_intercept), std::sqrt(delta_intercept));
}
std::vector<double> linreg::expected_x(const std::vector<double> &x, int steps){
    std::vector<double> u;
    if(x.size() < 1) return u;
    u = linreg::linspace(linreg::vector_min(x), linreg::vector_max(x), steps);
    return u;
}
std::vector<double> linreg::expected_y(const std::vector<double> &x, double slope, double intercept){
    std::vector<double> v;
    if(x.size() < 1) return v;
    double y_i = 0.;
    for(const double x_i : x){
        y_i = intercept + (slope*x_i);
        v.push_back(y_i);
    }
    return v;
}