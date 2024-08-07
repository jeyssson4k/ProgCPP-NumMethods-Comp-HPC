#include "integration.h"

int main(void){
    float y0 = trapezoid(0.f, 1.f, 10, f);
    float y1 = trapezoid(0.f, 1.f, 1000, f);
    float y2 = trapezoid(0.f, 1.f, 10000000, f);

    printf("%.10f\t%.10f\n%.10f\n", y0, y1, y2);

    return 0;
}