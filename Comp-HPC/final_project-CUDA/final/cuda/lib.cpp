#include "lib.h"

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

void setThreads(S *s, int threads_default)
{
    s->threadsPerBlock = (s->threadsPerBlock % 32 != 0 && s->k == 0) ? threads_default : s->threadsPerBlock;
}

void computeBlocks(S *s)
{
    s->blocksPerGrid = static_cast<int>((s->size + s->threadsPerBlock - 1) / s->threadsPerBlock);
}
void init(S *s, char **argv, int k)
{
    s->size = static_cast<size_t>(std::atoll(argv[1]));
    s->seed = static_cast<size_t>(std::atoll(argv[2]));
    s->threadsPerBlock = std::atoi(argv[3]);
    s->a = std::atof(argv[4]);
    s->b = std::atof(argv[5]);
    s->u = s->b - s->a;
    s->k = k;
    setThreads(s, DEFAULT_THREADS);
    computeBlocks(s);
}

void restoreData(float *host, char *path)
{
    std::ifstream data_file(path);
    float x = 0.f;
    int it = 0;
    while (data_file >> x)
    {
        host[it] = x;
        it++;
    }

    data_file.close();
}

float fabs_err(float x, float y)
{
    return std::fabs(1 - (x / y));
}