#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <stdio.h>
#include <cstdlib>
#include <cstdarg>
#include <chrono>

#define A 0.f
#define B 6.55f

bool validate_params(const int PARAMS, const int argc)
{
    const bool USER_PARAMS = PARAMS + 1 == argc;
    if (!USER_PARAMS)
    {
        printf("ERROR: Insufficient number of parameters for performance.\n\tExpected %d parameters.\n\tGiven %d parameters.\n\n",
               PARAMS, argc - 1);
        return false;
    }
    else
    {
        return true;
    }
}

void rprintf(const char *fmt...)
{
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0')
    {
        if (*fmt == 's')
        {
            const char *s = va_arg(args, const char *);
            printf("%s:\t", s);
        }
        else if (*fmt == 'f')
        {
            double d = va_arg(args, double);
            printf("%.10f\n", d);
        }
        ++fmt;
    }

    va_end(args);
}

struct RandGen
{
    RandGen() {}

    __device__ float operator()(int idx)
    {
        thrust::default_random_engine randEng(idx);
        thrust::uniform_real_distribution<float> uniDist(A, B);
        randEng.discard(idx);
        float x = uniDist(randEng);
        return std::exp(-1.0 * x * x);
    }
};

int main(int argc, char **argv)
{
    const bool EXEC = validate_params(2, argc);
    if (!EXEC)
        return EXIT_FAILURE;

    const int N = std::atoi(argv[1]);
    const float EXPECTED_VALUE = std::atof(argv[2]);

    // Transfer to device and compute the sum.
    auto t1 = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float> d_vec(N * N);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N * N),
        d_vec.begin(),
        RandGen());
    float x = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<float>());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Time performing device_vector mapping and reduction: " << ms_int.count() << "ms\n";

    float u0 = (B - A) * x / (N * N);
    float v = std::fabs(1 - (u0 / EXPECTED_VALUE));
    rprintf("sfsf", "Result", u0, "Error", v);
    return EXIT_SUCCESS;
}