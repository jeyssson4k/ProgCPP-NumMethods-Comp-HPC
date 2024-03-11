/**
* Program3 - Project Euler
* @author Jeysson4K
* @workshop 02
* @course PROG CPP - NUM METHODS
**/

#include <iostream>
#include <vector>
#include <cmath>

int get_fibonacci(int k);

int main(int argc, char **argv){
    int x = std::atoi(argv[1]);
    if(x < 2){
        x = 2;
    }
    int fibo_sum = get_fibonacci(x);
    std::cout<<fibo_sum<<"\n";
    return 0;
}

int get_fibonacci(int k){
    std::vector<int> fibonacci = {1,1};
    int sum = 1;
    int a = 2;
    while(sum <= k){
        int fib_term = fibonacci[a-1] + fibonacci[a-2];
       
        a++;
        fibonacci.push_back(fib_term);
        if(fib_term <= k){
            if(fib_term%2 == 0){
                continue;
            }
            sum += fib_term;
            //std::cout<<fib_term<<"\t"<<fibonacci[a-2]<<"\t"<<fibonacci[a-3]<<"\n";
        }else{
            break;
        }
    }
    return sum;
}