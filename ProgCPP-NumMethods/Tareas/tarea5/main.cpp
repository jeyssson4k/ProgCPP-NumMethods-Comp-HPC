#include "utils.hpp"

int main(int argc, char **argv){
    const std::string file_path = argv[1];
    const std::vector<std::vector<double>> data = linreg::fetch(file_path);
    const double e = 1.602176634e-19, h = 6.62607015e-34;

    auto [slope, delta_slope, intercept, delta_intercept, s] = linreg::linregress(data[0], data[1], true);

    std::vector<double> x = linreg::expected_x(data[0], 100);
    std::vector<double> y = linreg::expected_y(x, slope, intercept);

    std::printf("phi: %.5e\n", intercept);
    const double est_h = slope*e;
    linreg::show_error(est_h, h, "h");
    return EXIT_SUCCESS;
}