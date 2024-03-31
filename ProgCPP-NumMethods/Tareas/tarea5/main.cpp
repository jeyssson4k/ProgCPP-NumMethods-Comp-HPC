#include "utils.hpp"

int main(int argc, char **argv){
    const std::string file_path = argv[1];
    const std::vector<std::vector<double>> data = linreg::fetch(file_path);
    const double e = 1.602176634e-19, h = 6.62607015e-34;

    auto [slope, dslp, intercept, dint, s] = linreg::linregress(data[0], data[1], true);

    std::printf("phi: %.5e\n", intercept);
    const double est_h = slope*e;
    linreg::show_error(est_h, h, "h");

    const std::string dlm = "|";
    std::string 
        title    = "Tensión Eléctrica en función de la frecuencia de la luz incidente",
        xlabel   = "Frecuencia de la luz incidente",
        ylabel   = "Tensión Eléctrica",
        lr_label = "Regresión Lineal de V(v)",
        cv_label = "V(v)",
        lr_color = "#FF02B0",
        cv_color = "#000000";

    std::vector<double> v{slope, intercept, dslp, dint, est_h};
    std::string g_info = 
        "title,"    + title    + dlm +
        "xlabel,"   + xlabel   + dlm +
        "ylabel,"   + ylabel   + dlm +
        "lrlabel,"  + lr_label + dlm +
        "cvlabel,"  + cv_label + dlm +
        "lrcolor,"  + lr_color + dlm +
        "cvcolor,"  + cv_color ;
    
    linreg::export_linreg_values("out.txt", data[0], slope, intercept);
    linreg::export_JSON_linreg_info("info.json", g_info, "|", v);
    return EXIT_SUCCESS;
}