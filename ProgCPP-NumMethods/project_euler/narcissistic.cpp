#include <cmath>
#include <vector>
#include <cstdio>

std::vector<short> split_number(int number, short exp);
short base10_counter(int number);
bool narcissistic( int value );
int main(){
  std::vector<int> n{narcissistic(7), narcissistic(371), narcissistic(122), narcissistic(4887)};
	for(const int i : n){
    printf("%d, ",i);
  }
}
bool narcissistic( int value ){
  short base = base10_counter(value);
  std::vector<short> n = split_number(value, base);
  int nar = 0;
  for(const short i : n){
    printf("Esto es %d,%d,%f\n",i,base,std::pow(i,base));
    nar+=std::pow(i,base);
  }
  std::printf("%d es igual a %d?\n",value, nar);
  return nar == value;
}
short base10_counter(int number){
  int div = 10;
  short counter = 1;
  if(number < 10) return counter;
  while(true){
    if(number/div < 10){
      return ++counter;
    }else{
      div *= 10;
      counter++;
    }
  }
}
std::vector<short> split_number(int number, short exp){
  std::vector<short> number_as_vector {};
  int div = std::pow(10,exp-1);
  while(div >= 10){
    number_as_vector.push_back(number/div);
    number = number%div;
    div /= 10;
  }
  number_as_vector.push_back(number%10);

  return number_as_vector;
}