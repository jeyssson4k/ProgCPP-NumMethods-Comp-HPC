/**
* Program2 - Project Euler
* @author Jeysson4K
* @workshop 02
* @course PROG CPP - NUM METHODS
**/

#include <iostream>
#include <vector>
#include <cmath>

long get_primes(long k);
bool is_prime(long n);
int main(int argc, char **argv){long x = std::atol(argv[1]);
    long best_prime = get_primes(x);
    std::cout<<best_prime<<"\n\n";

    return 0;
}
long get_primes(long k){
    for(long a = 2; a <= k; a++){
        if(is_prime(a)){
            if(k%a == 0 and a <= k){
                if(k == a){
                    return k;
                }
                k = k / a ;
                a--;
            }
        } 
    }
    return k;
}

bool is_prime(long n){
    for(long b = 2; b <= std::sqrt(n); b++){
        if(n%b == 0){
            return false;
        }
    }
    return true;
}