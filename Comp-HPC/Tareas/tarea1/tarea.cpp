//============================================================================
// Name        : tarea02DiferenciaSumas.cpp
// Author      : Johan Alejandro López Arias
// Version     :
// Copyright   : Your copyright notice
// Description : Diferencias Numericas en Sumas
//============================================================================

#include <iostream> // Incluye la librería estándar de entrada y salida de C++
#include <iomanip>  // Incluye la librería para manipular la salida formateada, como setw y setprecision
#include <cmath>    // Incluye la librería matemática
#include <fstream> // Incluye la librería para operaciones de archivo

float sumaS1(int N);
float sumaS2(int N);

int main() { // Punto de entrada del programa
    const int MAX_N = 1000000; // Define el valor máximo de N como una constante de 1 millón

    std::ofstream archivo("sumatorias.csv"); // Crea y abre un archivo CSV para la escritura

    // Verifica si el archivo se abrió correctamente
    if (!archivo.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escribir." << std::endl;
        return 1;
    }

    // Escribe los encabezados en el archivo CSV
    archivo << "N;S1;S2;Delta\n";

    // Imprime los encabezados de la tabla
    //std::cout << std::setw(10) << "N" << std::setw(20) << "S1" << std::setw(20) << "S2";
    //std::cout << std::setw(20) << "Delta" << std::endl;

    // Calcular y mostrar los valores para diferentes N
    for(int N = 1; N <= MAX_N; ++N ) { // Bucle que incrementa N en potencias de 10 hasta MAX_N
        float S1 = sumaS1(N); // Calcula S1(N) usando la función sumaS1
        float S2 = sumaS2(N); // Calcula S2(N) usando la función sumaS2
        float delta = std::fabs(1.0f - S1 / S2); // Calcula la diferencia relativa Delta

        // Imprime la fila de la tabla
        std::cout << std::setw(10) << N << std::endl; // Imprime el valor de N
        //std::cout << std::setw(20) << std::setprecision(7) << S1; // Imprime S1 con 7 dígitos de precisión
        //std::cout << std::setw(20) << std::setprecision(7) << S2; // Imprime S2 con 7 dígitos de precisión
        //std::cout << std::setw(20) << std::setprecision(7) << delta << std::endl; // Imprime Delta con 7 dígitos de precisión
        // Escribe la fila actual en el archivo CSV
         archivo << N << ";" << std::setprecision(7) << S1 << ";";
         archivo << std::setprecision(7) << S2 << ";" << std::setprecision(7) << delta << "\n";

    }

    archivo.close(); // Cierra el archivo
    std::cout << "Archivo sumatorias.csv creado con éxito." << std::endl;

    return 0; // Termina el programa con código de éxito
}

// Función para calcular la suma S1
float sumaS1(int N) { // Define la función sumaS1 que toma un entero N y retorna un flotante
    float suma = 0.0f; // Inicializa la suma como un flotante a 0
    for(int k = 1; k <= N; ++k) { // Bucle desde k=1 hasta N incrementando k en cada iteración
        suma += 1.0f / k; // Acumula la suma de 1/k en la variable suma
    }
    return suma; // Retorna el valor final de la suma
}

// Función para calcular la suma S2
float sumaS2(int N) { // Define la función sumaS2 que toma un entero N y retorna un flotante
    float suma = 0.0f; // Inicializa la suma como un flotante a 0
    for(int k = N; k >= 1; --k) { // Bucle desde k=N hasta 1 decrementando k en cada iteración
        suma += 1.0f / k; // Acumula la suma de 1/k en la variable suma
    }
    return suma; // Retorna el valor final de la suma
}
