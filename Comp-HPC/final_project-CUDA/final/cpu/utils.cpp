#include "utils.h"

bool utils::validate_params(const int PARAMS, const int argc){
    const bool USER_PARAMS = PARAMS+1 == argc;
    if(!USER_PARAMS){
        printf("ERROR: Insufficient number of parameters for performance.\n\tExpected %d parameters.\n\tGiven %d parameters.\n\n", 
            PARAMS, argc-1);
        return false;
    }else{
        return true;
    }
}
 
void utils::rprintf(const char* fmt...){
    va_list args;
    va_start(args, fmt);
 
    while (*fmt != '\0'){
        if (*fmt == 's'){
            const char* s = va_arg(args, const char*);
            printf("%s:\t", s);
        }
        else if (*fmt == 'f'){
            double d = va_arg(args, double);
            printf("%.10f\n", d);
        }
        ++fmt;
    }

    va_end(args);
}

double utils::f(double& x){
    return std::exp(-1.0*x*x);
}