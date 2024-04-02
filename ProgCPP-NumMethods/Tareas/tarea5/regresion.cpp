#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdarg>
#include <tuple>
#include <limits>

template<typename T>
double map(T f_x, const std::vector<double> &v);
template<typename T>
double dual_map(T f_xy, const std::vector<double> &u, const std::vector<double> &v);
std::vector<std::vector<double>> fetch(const std::string &path);
void show(const std::vector<double> summary);
void show_error(const double x, const double y, std::string var);
double r_error(const double x, const double y);
double vector_max(const std::vector<double> &v);
double vector_min(const std::vector<double> &v);
std::vector<double> linspace(double a, double b, int steps);
std::tuple<double, double, double, double, double> linregress(const std::vector<double>& x, const std::vector<double>& y, bool show_results);
std::vector<double> expected_x(const std::vector<double> &x);
std::vector<double> expected_y(const std::vector<double> &x, double slope, double intercept);
void export_linreg_values(std::string path, const std::vector<double> &x, double slope, double intercept);

int main(int argc, char **argv){
    const std::string fname = argv[1]; //capturar el nombre del archivo
    const std::vector<std::vector<double>> data = fetch(fname); //separar los datos
    const double e = 1.602176634e-19, h = 6.62607015e-34;

    //calcular los datos de la regresión
    auto [slope, dslp, intercept, dint, s] = linregress(data[0], data[1], true);

    //mostrar datos 
    std::printf("phi: %.5e\n", intercept);
    const double est_h = slope*e;
    show_error(est_h, h, "h");
    //guardar los datos de la regresión para graficar
    export_linreg_values("out.txt", data[0], slope, intercept); 

    return EXIT_SUCCESS;
}

//hacer la suma respecto a una variable
template<typename T>
double map(T f_x, const std::vector<double> &v){
    double s = 0.;
    if(v.size() < 1) return s;
    for(const double v_i : v){
        s += f_x(v_i);
    }
    return s;
}
//hacer la suma respecto a dos variables
template<typename T>
double dual_map(T f_xy, const std::vector<double> &u, const std::vector<double> &v){
    double s = 0.;
    if(u.size() != v.size()){
        return s;
    }else{
        for(int i = 0; i < u.size(); ++i){
            s += f_xy(u[i], v[i]);
        }
        return s;
    }   
}
//leer los datos del archivo
std::vector<std::vector<double>> fetch(const std::string &path){
    std::vector<std::vector<double>> data;
    std::vector<double> u, v;
    std::ifstream data_file(path);
    double x = 0., y = 0.;
    int it = 0;
    while(data_file >> x >> y){
        u.push_back(x); v.push_back(y); it++;
    }
    if(it == 0){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Error: file not found or file is empty!" <<
            "\n\t u.size: " << v.size() <<
            "\n\t v.size: " << v.size() <<
            "\n\t Try to use a valid path. \n" <<
            "--------------------------------------------------------------------------------\n";
        data.push_back(u); data.push_back(v);
        return data;
    }
    data.push_back(u); data.push_back(v);
    data_file.close();
    std::cerr << 
        "--------------------------------------------------------------------------------\n" << 
        "Operation success: file converted to x,y vectors!" <<
        "\n\t x: " << it << " elements." <<
        "\n\t y: " << it << " elements.\n" <<
        "--------------------------------------------------------------------------------\n";
    return data;
}
//mostrar los resultados en consola
void show(const std::vector<double> summary){
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf(
        "Summary\n\t slope: %.5e +- %.5e\n\t intercept: %.5e +- %.5e\n\t s: %.5e \n\n", 
        summary[0], summary[1], summary[2], summary[3], summary[4]
    );
    std::printf("Linear Regression executed successfully!\n");
    std::printf("--------------------------------------------------------------------------------\n");
}
//hacer una comparación de un valor obtenido con respecto a su valor esperado
void show_error(const double x, const double y, std::string var){
    const double err = r_error(x,y);
    const char* char_txt = var.c_str();
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf(
        "Error summary for %s\n\t Expected value: %.16e\n\t Current value: %.16e\n\t Error: %.16e\n",char_txt, y, x, err
    );
    std::printf("--------------------------------------------------------------------------------\n");    
}
//calcular error relativo
double r_error(const double x, const double y){
    if(x == y) return 0.;
    if(std::fabs(y-std::numeric_limits<double>::min()) < 1e-300){
        return -1.; //error de precisión: los números están muy cerca del overflow
    }
    return std::fabs(1 - (x/y));
}
//calcular el valor máximo de un vector
double vector_max(const std::vector<double> &v){
    if(v.size() == 0) return 0.;
    double max = v[0];
    for(const double v_i : v){
        if(v_i > max){
            max = v_i;
        }
    }
    return max;
}
//calcular el valor mínimo de un vector
double vector_min(const std::vector<double> &v){
    if(v.size() == 0) return 0.;
    double min = v[0];
    for(const double v_i : v){
        if(v_i < min){
            min = v_i;
        }
    }
    return min;
}
//crear un vector homogeneamente espaciado
//este método actúa de forma similar a numpy.linspace en Python
std::vector<double> linspace(double a, double b, int steps){
    std::vector<double> w; 
    if(b<=a){
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
std::tuple<double, double, double, double, double> linregress(const std::vector<double>& x, const std::vector<double>& y, bool show_results){
    double N = 0., slope = 0., intercept = 0., s = 0., delta_slope = 0., delta_intercept = 0.;
    if(x.size()-y.size() == 0){
        N = static_cast<double>(x.size());
    }else{
        return std::make_tuple(slope, delta_slope, intercept, delta_intercept, s);
    }

    //lambdas que representan las maneras en que se hacen las sumas
    auto f = [](double x){return x;};
    auto g = [](double x){return x*x;};
    auto h = [](double x, double y){return x*y;};

    //calcular los términos 
    const double 
        k    = 1/N,
        e_x  = k*map(f, x),
        e_y  = k*map(f, y),
        e_xx = k*map(g, x),
        e_yy = k*map(g, y),
        e_xy = k*dual_map(h, x, y);
    
    if(r_error(e_xx, e_x) < 1e-4){
        //evitar una división que tienda a 0 
        return std::make_tuple(slope, delta_slope, intercept, delta_intercept, s);
    }

    //calcular los valores de la regresión lineal
    slope           = (e_xy - e_x*e_y)/(e_xx - (e_x*e_x));
    intercept       = ((e_xx*e_y) - (e_x*e_xy))/(e_xx - (e_x*e_x));
    s               = (N/(N-2))*(e_yy - (e_y*e_y) - (slope*slope)*(e_xx - (e_x*e_x)));
    delta_slope     = (s/(N*(e_xx - (e_x*e_x))));
    delta_intercept = (delta_slope*e_xx);

    //mostrar los resultados si es necesario
    if(show_results){
        show({slope, delta_slope, intercept, delta_intercept, s});
    }

    return std::make_tuple(slope, std::sqrt(delta_slope), intercept, std::sqrt(delta_intercept), std::sqrt(delta_intercept));
}
//estimar valores de x para la regresión
std::vector<double> expected_x(const std::vector<double> &x){
    std::vector<double> u;
    if(x.size() < 1) return u;
    u = linspace(vector_min(x), vector_max(x), x.size());
    return u;
}
//estimar valores de y a partir de los valores de x estimados 
std::vector<double> expected_y(const std::vector<double> &x, double slope, double intercept){
    std::vector<double> v;
    if(x.size() < 1) return v;
    double y_i = 0.;
    for(const double x_i : x){
        y_i = (slope*x_i) + intercept;
        v.push_back(y_i);
    }
    return v;
}
void export_linreg_values(std::string path, const std::vector<double> &x, double slope, double intercept){
    if(x.size() < 1){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export linear regression]: error processing data." <<
            "\n\t Vector x is empty!." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
    }else if(path.empty()){
        std::cerr << 
            "--------------------------------------------------------------------------------\n" << 
            "Argument Error [in export linear regression]: error processing data." <<
            "\n\t Path is empty!." <<
            "\n\t Try to use a valid input. \n" <<
            "--------------------------------------------------------------------------------\n";
    }else{
        std::ofstream outputFile(path);
        outputFile << std::setprecision(20) << std::scientific;
        std::vector<double> lr_x = expected_x(x);
        std::vector<double> lr_y = expected_y(lr_x, slope, intercept);

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
/*
Biblografía
https://en.cppreference.com/w/cpp/io/basic_ifstream
https://en.cppreference.com/w/cpp/container/vector
https://en.cppreference.com/w/cpp/types/numeric_limits
https://en.cppreference.com/w/cpp/utility/tuple
*/
