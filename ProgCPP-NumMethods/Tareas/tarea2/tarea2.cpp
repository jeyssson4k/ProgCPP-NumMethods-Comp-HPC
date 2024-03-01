#include <iostream>

int Fibonacci_sum(int n) {
    int a = 1;
    int b = 2;
    int suma = 0;

    while (a <= n) {
        if (a % 2 != 0) {
            suma += a;
        }
        int temp = a + b;
        a = b;
        b = temp;
    }

    return suma;
}

int main() {
    while (true) {
        std::cout << "Ingrese un n�mero (debe ser mayor o igual a 2, ingrese un n�mero menor a 2 para salir): ";
        int n;
        std::cin >> n;

        if (n < 2) {
            std::cout << "Saliendo del programa." << std::endl;
            break;
        }
        int suma = Fibonacci_sum(n);
        std::cout << "La suma de los t�rminos impares de la secuencia de Fibonacci hasta " << n << " es: " << suma << std::endl;
    }
    return 0;
}

