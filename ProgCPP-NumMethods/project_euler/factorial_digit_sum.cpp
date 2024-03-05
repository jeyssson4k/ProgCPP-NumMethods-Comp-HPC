#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

short base10_counter(long double number);
std::vector<short> split_number( long double number, short exp);
long double factorial(int n);
int sum_vector_components(std::vector<short> vector);

int main() {
  
  int n = 100;
  long double n_factorial = factorial(n);
  //std::printf("n_factorial: %LF\n\n", n_factorial);
  short base10 = base10_counter(n_factorial);
  //std::printf("%d\n", base10);
  std::vector<short> n_factorial_as_vector = split_number(n_factorial, base10);
  int sum = sum_vector_components(n_factorial_as_vector);

  std::vector<short> num{9,3,3,2,6,2,1,5,4,4,3,9,4,4,1,5,2,6,8,1,6,9,9,2,3,8,8,5,6,2,6,6,7,0,0,4,9,0,7,1,5,9,6,8,2,6,4,3,8,1,6,2,1,4,6,8,5,9,2,9,6,3,8,9,5,2,1,
  7,5,9,9,9,9,3,2,2,9,9,1,5,6,0,8,9,4,1,4,6,3,9,7,6,1,5,6,5,1,8,2,8,6,2,5,3,6,9,7,9,2,0,8,2,7,2,2,3,7,5,8,2,5,1,1,8,5,2,1,0,9,1,6,8,6,4};
  int sum2 = sum_vector_components(num);
  std::printf("Factorial: %Lf\tSuma: %d\tSuma Python: %d\n",n_factorial, sum, sum2);
  return 0;
}

long double factorial (int n){
    if(n == 1){
        return 1;
    }else{
        return (n*factorial(n-1));
    }
}

short base10_counter(long double number){
  long double div = 10;
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
std::vector<short> split_number(long double number, short exp){
  std::vector<short> number_as_vector {};
  //long double div = std::pow(10,exp);
  long double div = 1;
  for(int i = 0; i <=exp ; i++ ){
    div *= 10;
  }
  std::printf("div: %Lf\n", div);
  while(div >= 10){
    number_as_vector.push_back(number/div);
    number = std::fmod(number,div);
    div /= 10;
  }
  number_as_vector.push_back(std::fmod(number,10));

  return number_as_vector;
}

int sum_vector_components(std::vector<short> vector){
    int sum = 0;
    for(const short a : vector){
        sum+=a;
    }
    return sum;
}