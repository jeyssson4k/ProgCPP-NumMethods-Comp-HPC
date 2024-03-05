#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>

std::vector<int> nth_twin_primes(int n);
bool is_prime(int n);

int main(){
    int N = 100;
    for(int i = 1; i <= N; i++){
        std::vector<int> primes = nth_twin_primes(i);
        double norm = std::sqrt(primes[0]*primes[0] + primes[1]*primes[1]);
        std::printf("%d\t%.6f\n", i, norm);
    }
    return 0;
}

std::vector<int> nth_twin_primes(int n){
    int i = 0; //k = 0, j = 0;
    std::vector<int> values = {0,0};
    for(int h = 2; h < 1'000'000; h++){
        if(i == n) break;
        if(is_prime(h) and is_prime(h+2)){
            values[0] = h;
            values[1] = h+2;
            i++;
        }
    }
    //std::printf("(%d,%d)",values[0],values[1]);
    return values;
}

bool is_prime(int n){
    bool number_is_prime = true;
    for(int b = 2; b <= std::sqrt(n); b++){
        if(n%b == 0){
            number_is_prime = false;
            break;
        }
    }
    return number_is_prime;
}