//imprimir los numeros primos del 10 al 30
#include <iostream>

bool validarPrimo(short numero);
int main(){
    bool primo = false; 
    for(short i = 10; i <= 30; i++){
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
