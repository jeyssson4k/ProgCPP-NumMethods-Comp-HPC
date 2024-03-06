#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

bool is_prime(long n);
short base10_counter(int number);
bool is_prime_left(long n);
bool is_prime_right(long n);
int sum_truncatable_primes(std::vector<int> w);
int main(){
    short i = 0;
    long x = 211;
    std::vector<long> truncatable_primes {};
    while(1 <= 11){
        if(x/100 == 1 or x/1000 == 1 or x/10000 == 1){
            x++;
            continue;
        }
        std::printf("Estamos en x: %ld, llevamos %d primos\n", x, i);
        bool x_is_prime = is_prime(x);
        if(x_is_prime){
            bool x_has_primes_left = is_prime_left(x);
            if(x_has_primes_left){
                bool x_has_primes_right = is_prime_right(x);
                if(x_has_primes_right){
                    i++;
                    std::printf("Este es un truncatable prime: %ld\n",x);
                    truncatable_primes.push_back(x);
                    x++;
                    continue;
                }else{
                    x++;
                    continue;
                }
            }else{
                x++;
                continue;
            }
        }else{
            x++;
        }
    }
    
    for(const long prime : truncatable_primes){
        std::printf("%ld\t",prime);
    }
    return 0;
}

bool is_prime(long n){
    for(long b = 2; b <= std::sqrt(n); b++){
        if(n%b == 0 or n <= 1){
            return false;
        }
    }
    return true;
}
short base10_counter(int number){
  int div = 10;
  short counter = 1;
  while(true){
    if(number/div < 10){
      return counter;
    }else{
      div *= 10;
      counter++;
    }
  }
}
bool is_prime_left(long n){
    short base = base10_counter(n);
    long mod = 0;
    long div = std::pow(10,base);
    bool is_truncatable_prime = true;
    while(n>10){
        mod = static_cast<long>(std::fmod(n,div));
        //std::printf("Estoy en mod: %ld\n",mod);
        bool mod_is_prime = is_prime(mod);
        if(mod_is_prime){
            n = mod;
            div/=10;
        }else{
            is_truncatable_prime = false;
            break;
        }
    }
    return is_truncatable_prime;
}
bool is_prime_right(long n){
    long mod = 0;
    bool is_truncatable_prime = true;
    while(n>10){
        mod = n/10;
        std::printf("Esto es mod: %ld\n",mod);
        bool mod_is_prime = is_prime(mod);
        if(mod_is_prime){
            n = mod;
        }else{
            is_truncatable_prime = false;
            break;
        }
    }
    return is_truncatable_prime;
}
int sum_truncatable_primes(std::vector<int> w);