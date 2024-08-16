#include <iostream>

class Quaternion{
    public:
        double a{0.},b{0.},c{0.},d{0.};
        Quaternion(double a, double b, double c, double d){
            this->a = a;
            this->b = b;
            this->c = c;
            this->d = d;
        }
        Quaternion(){}
        Quaternion operator+(const Quaternion &q){
            return Quaternion{this->a+q.a, this->b+q.b, this->c+q.c, this->d+q.d};
        }
        Quaternion operator-(const Quaternion &q){
            return Quaternion{this->a-q.a, this->b-q.b, this->c-q.c, this->d-q.d};
        }
        Quaternion operator*(const Quaternion &q){
            return Quaternion{
                this->a*q.a - this->b*q.b - this->c*q.c - this->d*q.d, 
                this->a*q.b + this->b*q.a + this->c*q.d - this->d*q.c,
                this->a*q.c - this->b*q.d + this->c*q.a + this->d*q.b,
                this->a*q.d + this->b*q.c - this->c*q.b + this->d*q.a
            };
        }
        void print(){
            std::cout<<this->a<<"\t"<<this->b<<"\t"<<this->c<<"\t"<<this->d<<"\n";
        }
};

int main(void){
    Quaternion q1{1., 9.02, -0.76, 0.0043}, q2{0.09, 9.782, -5.47, 1.875}, q3;
    q3 = q1+q2;
    q3.print();
    q3 = q1-q2;
    q3.print();
    q3 = q1*q2;
    q3.print();
    return EXIT_SUCCESS;
}