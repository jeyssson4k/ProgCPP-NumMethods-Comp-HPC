#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdio>

float f(int k);
float S1(int N);
float S2(int N);

int main(){
    int N = 2'500'000;
    const std::string output = "out.txt";

    std::ofstream outputFile(output);

    outputFile << std::fixed << std::setprecision(16);
    for(int i = 1; i <= N; i+=250){
        std::cout.precision(16);
        float s_1 = S1(i);
        float s_2 = S2(i);
        float delta = std::abs(1-(s_1/s_2));

        outputFile <<i<<";"<<delta<<"\n";
        
        std::printf("N: %.16f\t S1: %.16f\t S2: %.16f\t Error: %.16f \n", i,s_1, s_2, delta);
        
    }
    
    outputFile.close();
    return 0;
}

float f(int k){
    return 1.00/(k+0.00f);
}
float S1(int N){
    float S = 0.00f;
    for (int i = 1; i <= N; i++){
        S += f(i);
    }
    return S;
}
float S2(int N){
    float S = 0.00f;
    for (int i = N; i >= 1; i--){
        S += f(i);
    }
    return S;
}