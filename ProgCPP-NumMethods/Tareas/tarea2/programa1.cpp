/**
* Program1 - Project Euler
* @author Jeysson4K
* @workshop 02
* @course PROG CPP - NUM METHODS
**/

#include <iostream>
#include <vector>
#include <cmath>

std::vector<int> get_primes(int k);
bool is_prime(int n);
long long sum_primes(std::vector<int> primes);

int main(int argc, char **argv){
    int x = std::atoi(argv[1]);
    std::vector<int> primes_less_than_x = get_primes(x);
    long long sum = sum_primes(primes_less_than_x);
    std::cout<<sum<<"\n";
    return 0;
}

std::vector<int> get_primes(int k){
    std::vector<int> primes;
    for(int a = 2; a < k; a++){
        bool a_is_prime = is_prime(a);
        if(a_is_prime){
            primes.push_back(a);
        }
    }
    return primes;
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
long long sum_primes(std::vector<int> primes){
    long long sum = 0;
    for(int prime : primes){
        sum += prime;
    }
    return sum;
}