#include <iostream>
#include <vector>
#include <array>
#include <cmath>

int main(){
    int bound = 57;
    double sum = 0.0;
    int i = 1;
    while(i<=bound){
        sum += 1/i;
        i+=1;
    }
    std::cout << sum;

    return 0;
}