#include <iostream>
#include <cmath>

double pi_aprox (int n);
int main(void)
{
    double x= M_PI;
    std::cout <<"El Valor de pi es " << "\n" << x << "\n";
    std::cout.precision(16);
    std::cout.setf(std::ios::scientific);
    int nmin=1, nmax=20;
    std::cout <<"valor aproximación"<< "\n";
    for(int n=nmin; n<=nmax; n = n+1)
    {
    std::cout << pi_aprox (n)<< "\n";
   
    }
     std::cout <<"valor de la diferencia"<< "\n";
    for(int n=nmin; n<=nmax; n = n+1)
    {
       
        std::cout <<(1-(pi_aprox(n)/M_PI))<< "\n";
       
    }  

    return 0;
}
double pi_aprox (int n)
{
    double suma;
    double r=0;
    double s=0;
    for (double r=0;r<n+1;r++)
    {
     double elevado=std::pow (16,r);
    suma=((1/(elevado))*((4/(8*r+1))-(2/(8*r+4))-(1/(8*r+5))-(1/(8*r+6))));

    s=s+suma;
    }
    return s;
}