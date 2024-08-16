#include <iostream>
#include <vector>

#define g 9.81
#define dt 0.01
#define t0 0.0
#define tf 2.3456
#define steps (int) (tf-t0)/dt
class Particle{
    double m{1.}, Rz{0.}, Vz{0.}, Fz{0.};
};

typedef std::vector<Particle> Particles;
void start(Particles &p);
void compute_forces(Particles &p);

int main(void){
    int N = 1;
    Particles p0(N);
    start(p0);
    compute_forces(p0);

    for(int i=0; i <= steps; ++i){
        
    }
    return EXIT_SUCCESS;
}
void start(Particles &p){
    for(auto &p_i : p){
        p_i.Rz = 1.2345678;
        p_i.Vz = 3.2325223;
    }
}
void start(Particles &p){
    for(auto &p_i : p){
        p_i.Fz = 0.0;
    }
    for(auto &p_i : p){
        p_i.Fz -= p_i.m*g;
    }
}