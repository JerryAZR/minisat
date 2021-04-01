#ifndef __CUDASOLVER
#define __CUDASOLVER

#include "minisat/core/SolverTypes.h"

#define CL_NOCHANGE 0
#define CL_NEWBLOCK 1
#define CL_NEWWATCH 2
#define CL_UNIT     3
#define CL_CONFLICT 4

#define VAR(x) (x >> 1)
#define SIGN(x) (x & 1)
#define VALUE(x, assigns) (assigns[x >> 1] ^ (x & 1))

#define LT ((uint8_t)0)
#define LF ((uint8_t)1)
#define LU ((uint8_t)2)

#define CREF_UNDEF  0xFFFFFFFF
#define LIT_UNDEF   0xFFFFFFFF

#define USE_CUDA
#define CHECK_CUDA_ERROR

#define MAX_CONFL   256

__global__ void checkConflict(int* clauses, unsigned* ends, unsigned* crefs,
    unsigned clauseCount, uint8_t* assigns, int* lock, unsigned* conflicts, unsigned* conflCount,
    int* implications, unsigned* implSource, unsigned* implCount);

// test functions
void checkCudaError(const char msg[]);

#endif
