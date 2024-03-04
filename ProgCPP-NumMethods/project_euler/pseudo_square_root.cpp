/*
The divisors of 12 are: 1,2,3,4,6 and 12.
The largest divisor of 12 that does not exceed the square root of 12 is 3.
We shall call the largest divisor of an integer n that does not exceed the square root of n the pseudo square root (PSR) of n.
It can be seen that PSR(3102)=47.
Let p be the product of the primes below 190.
Find PSR(p) mod 10^{16}.
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

long double PSR(long double n);
std::vector<long> get_primes(long k);
long double calculate_product_of_primes(std::vector<long> primes);
bool is_prime(long n);

int main() {
  long double div = 10'000'000'000'000'000;
  int x_0 = 190;
  long double p = calculate_product_of_primes(get_primes(x_0));
  long double psr = PSR(p);
  std::printf("product: %Lf, psr: %Lf\n", p, psr);

  long double P_x = std::fmod(psr,div);
  std::printf("P_x: %Lf\n", P_x);
  return 0;
}

long double PSR(long double n) {
  long double pseudo_square_root = 1.0, max = std::floor(std::sqrt(n))-1;
  for(long double i = max; i > 2.0; i--){
    long double x = n/i;
    std::printf("x: %.1Lf\n",x);
    if(x == 0.00){
      pseudo_square_root = i;
      break;
    }
  }
  while(true){
    p = (max+2.0)/2.0;
  }
  return pseudo_square_root;
}

long double calculate_product_of_primes(std::vector<long> primes){
  long double product = 1;

  for(const long prime : primes){
    //std::printf("Prime: %ld, Product: %Lf\n",prime, product);
    product *= prime;
  }
  return product;
}
std::vector<long> get_primes(long k){
  std::vector<long> primes;
  for(long a = 2; a < k; a++){
    bool a_is_prime = is_prime(a);
    if(a_is_prime){
      primes.push_back(a);
    }
  }
  return primes;
}

bool is_prime(long n){
  bool number_is_prime = true;
  for(long b = 2; b <= std::sqrt(n); b++){
    if(n%b == 0){
      number_is_prime = false;
      break;
    }
  }
  return number_is_prime;
}
