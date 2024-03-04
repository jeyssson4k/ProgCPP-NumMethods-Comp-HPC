/*
A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 x 99.
Find the largest palindrome made from the product of two 3-digit numbers.
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

short base10_counter(int number);
std::vector<short> split_number(int number, short exp);
bool is_palindromic_number(int number, std::vector<short> number_vector, short exp);
std::vector<int> create_vector(int vi, int vf);
int main() {
  std::vector<int> u1 = create_vector(100,999);
  int largest_palindromic_product = 0;
  for(int n1 = u1[u1.size()-1];n1 >= u1[0];n1--){
    for(int n2 = u1[u1.size()-1];n2 >= u1[0];n2--){
      if(n2>n1) continue;
      int product = n1*n2;
      short base10 = base10_counter(product);
      std::vector<short> number_vector = split_number(product, base10);
      bool palindromic = is_palindromic_number(product, number_vector, base10);
      if(palindromic and product>largest_palindromic_product){
        largest_palindromic_product = product;
      }
    }
  }
  std::printf("The largest palindromic product between two 3-digit numbers is %d",largest_palindromic_product);
  return 0;
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
std::vector<short> split_number(int number, short exp){
  std::vector<short> number_as_vector {};
  int div = std::pow(10,exp);
  while(div >= 10){
    number_as_vector.push_back(number/div);
    number = number%div;
    div /= 10;
  }
  number_as_vector.push_back(number%10);

  return number_as_vector;
}
bool is_palindromic_number(int number, std::vector<short> number_vector, short exp){
  int palindromic = 0;
  for(int i = exp+1; i >= 0; i--){
    palindromic += number_vector[i]*std::pow(10, (i));
  }
  //std::printf("Es %d igual a %d?\n", number, palindromic);
  return palindromic == number;
}
std::vector<int> create_vector(int vi, int vf){
  std::vector<int> u {};
  for(int i = vi; i <= vf; i++){
    if(i%10 == 0) continue;
    u.push_back(i);
  }
  return u;
}

void tests(){
  const std::vector<int> numbers{11,121,5555,12321, 76};
  std::vector<bool> palindromic {};
  for(const auto number : numbers){
    short base10 = base10_counter(number);
    std::printf("Number: %d, Base10: 10+e%d, ", number, base10);
    std::vector<short> number_vector = split_number(number, base10);
    std::printf("Vector:{");
    for(const auto n : number_vector){
      std::printf(" %d",n);
    }
    std::printf(" }\n");
    palindromic.push_back(is_palindromic_number(number, number_vector,base10));
  }
  for(const auto p : palindromic){
    std::printf("%d ", static_cast<int>(p));
  }
  std::printf("\n\n\n");
}