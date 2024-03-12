#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

double f(double x); 
double df_dx(double x);
double deriv_forward(double x, double h, bool is_central);
double df_error(double dy_theo, double dy_num);

int main(void){
    //sets the specific value for x to be used
    const double x = 4.321;
    //puts into a vector all values of "h" to be used
    const std::vector<double> h {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};
    //creates an instance of fstream to write in a file
    const std::string output_data_file_name = "data.txt";
    std::ofstream outputFile(output_data_file_name);
    outputFile << std::fixed << std::setprecision(20);
    
    //for each value in h vector, derivates f and calcutates delta
    for(const double h_0 : h){
        //theorical value of df/dx
        const double df = df_dx(x);
        //numerical values of df/dx
        const double df_forward = deriv_forward(x,h_0,false);
        const double df_central = deriv_forward(x,h_0,true);
        //write in file
        outputFile <<h_0<<"\t"<<df_error(df,df_forward)<<"\t"<<df_error(df,df_central)<<"\n";

        //optional: if you wanna see the output by console
        std::printf("%.20f\t%.20f\t%.20f\n",h_0, df_error(df,df_forward), df_error(df,df_central));

    }
    outputFile.close();
    return EXIT_SUCCESS;
}

//function to be derivated: 4xsin(x)+7
double f(double x){
    return (4*x*std::sin(x))+7;
}
//theorical derivate of function f
double df_dx(double x){
    return 4*(x*std::cos(x)+std::sin(x));
}
//numerical derivate of function f
double deriv_forward(double x, double h, bool is_central){
    if(is_central){
        return (f(x+(h/2.0))-f(x-(h/2.0)))/h; 
    }else{
        return (f(x+h)-f(x))/h;
    }
}
double df_error(double dy_theo, double dy_num){
    return std::fabs(dy_theo-dy_num)/std::fabs(dy_theo);
}
/*
double deriv_forward(double x, double h, bool is_central){
    return (is_central) ? ((f(x+(h/2.0))-f(x-(h/2.0)))/h) : ((f(x+h)-f(x))/h);
}
*/