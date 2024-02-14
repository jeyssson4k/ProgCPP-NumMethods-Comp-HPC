//imprimir los primeros 100 numeros primos 
#include <iostream>

bool validarPrimo(short numero);
int main(){
    bool primo = false; 
    for(short i = 2; i <= 100; i++){
        primo = validarPrimo(i);
        if(primo){
            std::cout<<i<<std::endl;
        }
    }
    return 0;
}

bool validarPrimo(short numero){
    bool estado = true;
    for(short divisor = 2; divisor < numero; divisor++){
        if(numero%divisor == 0){
            estado = false;
            break;
        }
    }
    return estado;
}
