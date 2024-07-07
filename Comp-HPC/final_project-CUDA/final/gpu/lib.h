#pragma once
#include <stdio.h>
#include <cstdlib>
#include <cstdarg>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <time.h>

#define CUDA_CALL(x)                                                                     \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = (x);                                                           \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                                         \
        }                                                                                \
    } while (0)

#define CURAND_CALL(x)                                      \
    do                                                      \
    {                                                       \
        if ((x) != CURAND_STATUS_SUCCESS)                   \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

#define BYTES sizeof(float)
#define SHARED_MEMORY 256 * BYTES
#define DEFAULT_THREADS 256
#define MAX_THREADS 1024

struct S
{
    size_t size;
    size_t seed;
    int threadsPerBlock;
    size_t blocksPerGrid;
    float a;
    float b;
    float u;
    int k;

    S() : size(0), seed(0), blocksPerGrid(0), threadsPerBlock(0), a(0.0f), b(0.0f), u(0.0f), k(0) {}
};

struct D
{
    float *d_i;
    float *d_o;
    float *d_o2;
};

struct H
{
    float *h_o;
};
enum K
{
    CUDA_REDUCTION,
    CUDA_INTEGRATION
};

bool validate_params(const int PARAMS, const int argc);
void rprintf(const char *fmt...);
void setThreads(S *, int);
void computeBlocks(S *);
void init(S *, char **, int);
void restoreData(float *host, char *path);
float fabs_err(float, float);