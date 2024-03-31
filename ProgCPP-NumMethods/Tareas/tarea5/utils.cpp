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
std::vector<double> linreg::expected_x(const std::vector<double> &x){
    std::vector<double> u;
    if(x.size() < 1) return u;
    u = linreg::linspace(linreg::vector_min(x), linreg::vector_max(x), x.size());
    return u;
}
std::vector<double> linreg::expected_y(const std::vector<double> &x, double slope, double intercept){
    std::vector<double> v;
    if(x.size() < 1) return v;
    double y_i = 0.;
    for(const double x_i : x){
        y_i = (slope*x_i) + intercept;
        v.push_back(y_i);
    }
    return v;
}
void linreg::export_linreg_values(std::string path, const std::vector<double> &x, double slope, double intercept){
    if(x.size() < 1){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export linear regression]: error processing data to export the linear regression." <<
            "\n\t Vector x is empty!." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
    }else if(path.empty()){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export linear regression]: error processing data to export the linear regression." <<
            "\n\t Path is empty!." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
    }else{
        std::ofstream outputFile(path);
        outputFile << std::setprecision(20) << std::scientific;
        std::vector<double> lr_x = linreg::expected_x(x);
        std::vector<double> lr_y = linreg::expected_y(lr_x, slope, intercept);

        for(int i = 0; i < x.size(); ++i){
            outputFile << lr_x[i] << "\t" << lr_y[i] << "\n";
        }

        outputFile.close();
        std::cout << 
            "--------------------------------------------------------------------------------\n" << 
            "Operation completed succesfully [in export linear regression]: Data written in " << path <<
            "\n\t Column 0: expected values of x." <<
            "\n\t Column 1: expected values of f(x) = mx+b.\n" <<
            "--------------------------------------------------------------------------------\n";
    }
        
}

void linreg::export_JSON_linreg_info(std::string path, std::string buffer, std::string delimiter, std::vector<double> stats){
    if(path.empty() or buffer.empty()){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export JSON info]: error processing data to export the linear regression." <<
            "\n\t Path is empty or no info to be processed!." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
    }else if(stats.size() < 4 or stats.size() > 5){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export JSON info]: error processing data to export the linear regression." <<
            "\n\t Some values are missing! (or too much parameters)." <<
            "\n\t Try to use a valid input. Verify if your vector has (at least):\n" <<
            "\n\t\t slope, intercept, delta_slope, delta_slope, delta_intercept (in this order) and try again.\n" <<
            "--------------------------------------------------------------------------------\n";
    }else{
        std::vector<std::string> v{"slope", "intercept", "dslp", "dint", "k"};
        std::ofstream outputFile(path);
        outputFile << std::setprecision(20) << std::scientific;
        outputFile << "{\n\t";
        for(int i = 0; i < stats.size(); ++i){
            outputFile << "\"" << v[i] << "\" : " << stats[i] << ",\n\t";
        }
        int splits = 0;
        std::size_t f = buffer.find(delimiter), g = buffer.find(",");
        while(f != std::string::npos or g != std::string::npos){
            std::size_t found = buffer.find(delimiter);
            if(found!=std::string::npos){
                std::string str = buffer.substr(0,found);
                buffer.erase(0,found+1); 
                std::size_t fnd = str.find(",");
                if(fnd!=std::string::npos){
                    std::string s1 = str.substr(0,fnd);
                    str.erase(0,fnd+1); 
                    std::string s2 = str;
                    str.erase(0, str.length());
                    std::cout << "\n\t" << "\"" << s1 << "\" : " << s2;
                    outputFile << "\"" << s1 << "\" : \"" << s2 << "\",\n\t";
                    splits++;
                    f = buffer.find(delimiter); g = buffer.find(",");
                    continue;
                }else{
                    std::cerr << 
                        "--------------------------------------------------------------------------------\n" << 
                        "Argument Error [in export JSON info]: error processing data to export the linear regression." <<
                        "\n\t Expected separator , in (key,value) #" << splits << "."
                        "\n\t Try to use a valid input. JSON will be unfinished at " << path << "\n" <<
                        "--------------------------------------------------------------------------------\n";
                    break;
                }
            }else{
                std::size_t fnd = buffer.find(",");
                if(fnd!=std::string::npos){
                    std::string s1 = buffer.substr(0,fnd);
                    buffer.erase(0,fnd+1); 
                    std::string s2 = buffer;
                    buffer.erase(0, buffer.length());
                    std::cout << "\n\t" << "\"" << s1 << "\" : " << s2 << "\n\n";
                    outputFile << "\"" << s1 << "\" : \"" << s2 << "\"\n\t";
                    f = buffer.find(delimiter); g = buffer.find(",");
                    splits++;
                    continue;
                }else{
                    std::cerr << 
                        "--------------------------------------------------------------------------------\n" << 
                        "Argument Error [in export JSON info]: error processing data to export the linear regression." <<
                        "\n\t Expected separator , in (key,value) #" << splits << "."
                        "\n\t Try to use a valid input. JSON will be unfinished at " << path << "\n" <<
                        "--------------------------------------------------------------------------------\n";
                    break;
                }
            }
            
        }

        outputFile << "\n}";
        outputFile.close();

        std::cout << 
            "--------------------------------------------------------------------------------\n" << 
            "Operation completed succesfully [in export JSON info]: " << splits + 1 <<
            " pairs (key,value) written in " << path << "\n" <<
            "--------------------------------------------------------------------------------\n";
    }
    
}

