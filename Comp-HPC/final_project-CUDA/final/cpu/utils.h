#pragma once

#include <stdio.h>
#include <cstdlib>
#include <cstdarg>
#include <random>
#include <vector>
#include <omp.h>

namespace utils
{
    bool validate_params(const int PARAMS, const int argc);
    void rprintf(const char *fmt...);
    double f(double &x);
    
    class MontecarloIO
    {
    private:
        std::random_device mc;
        std::mt19937 rng;
        std::uniform_real_distribution<double> random_x_i;
        double a;
        double b;
        size_t N;
        double sum;
        double res;
        double err;
    public:
        MontecarloIO(double a, double b, size_t N)
        {
            this->a = a;
            this->b = b;
            this->N = N;
            this->sum = 0.0;
            this->res = 0.0;
            this->err = 0.0;
        }
        template <typename T>
        void montecarlo(T f)
        {
            printf("Executing using single core...\n");
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<double> dist(this->a, this->b);
            this->sum = 0.0;
            for (size_t i = 0ULL; i < this->N; i++)
            {
                double r = dist(rng);
                this->sum += f(r);
            }
        }
        template <typename T>
        void montecarlo_OMP(T f)
        {
            printf("Executing using multi core...\n");
            #pragma omp parallel
            {
                int num_threads = omp_get_num_threads(),
                    thread_id   = omp_get_thread_num();
                if (thread_id == 0) {
                    printf("Number of threads = %d\n", num_threads);
                } 
            }

            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<double> dist(this->a, this->b);
            this->sum = 0.0;
            double s = 0.0;
            #pragma omp parallel for reduction(+ : s)
            for (size_t i = 0ULL; i < this->N; i++)
            {
                double r = dist(rng);
                s += f(r);
            }
            this->sum = s;
        }
        void computeResult()
        {
            this->res = (this->b - this->a) * this->sum / this->N;
        }
        void computeError(const double &EXPECTED_VALUE)
        {
            this->err = std::fabs(1 - (this->res / EXPECTED_VALUE));
        }
        double getRes()
        {
            return this->res;
        }
        double getError()
        {
            return this->err;
        }
    };
}
