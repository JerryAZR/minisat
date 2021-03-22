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

int checkWatcher(int blocker, int p, int* c, unsigned c_size, uint8_t* assigns);
__global__ void propagateKernel(int clauseCount, int* clauses, int* clausesEnd, uint8_t* assigns, int* actions);

#endif
