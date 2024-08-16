#include <iostream>
#include <vector>
#include <array>
#include <cmath>

int main(){
    std::vector<int> v= {1,2,3,4,5};
    int arr[] = {1,2,3,4,5};
    std::array<int, 5> a = {1,2,3,4,5};

    std::cout << v.size() << " " << sizeof(arr)/sizeof(arr[0]) << " " << a.size();
    std::cout << std::assoc_laguerre(0,10,0.5);
    return 0;
}