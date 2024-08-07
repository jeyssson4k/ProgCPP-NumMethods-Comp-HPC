#pragma once
#include <stdio.h>
#include <cstdlib>
#define function float 
template<typename T>
float trapezoid(float a, float b, size_t n, T f){
    float s = (f(a)/2.f) + (f(b)/2.f);
    float dx = (b-a)/((float)n);
    // Unfortunately, =+ is an actual operator 
    for(size_t i = 1; i < n; ++i)
        s += f(a+dx*i); 
    return dx*s;
}

function f(float x);