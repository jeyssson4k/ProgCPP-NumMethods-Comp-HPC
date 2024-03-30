#include "utils.hpp"

int main(int argc, char **argv){
    const std::string file_path = argv[1];
    const std::vector<std::vector<double>> data = linreg::fetch(file_path);
    const double N = static_cast<double>(data[0].size());

    const std::vector<double> a{1.,2.,3.};
    const std::vector<double> b{2.,3.,4.,5.};

    auto f = [](double x){return x;};
    auto g = [](double x){return x*x;};
    auto h = [](double x, double y){return x*y;};

    double xh = linreg::dual_map(h, a, b);
    double y = linreg::map(f,data[0]);
    std::printf("El valor es %.5f\n",y);
    std::printf("Hay %ld elementos en el vector\n", data[0].size());

    return EXIT_SUCCESS;
}