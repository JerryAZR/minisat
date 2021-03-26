#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"

void checkCudaError(const char msg[]) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("Error message: %s\n", msg);
        exit(1);
    }
}

void Minisat::Solver::verifyUnsat(CRef cr) {
    if (cr == CREF_UNDEF) {
        if (cpuCheckConflict()) {
            printf("Miss unsat.\n");
            exit(1);
        }
    } else {
        Clause& c = ca[cr];
        for (int i = 0; i < c.size(); i++) {
            if (value(c[i]) != l_False) {
                printf("False unsat.\n");
                exit(1);
            }
        }
    }
}

bool Minisat::Solver::cpuCheckConflict() {
    for (int i = 0; i < clauses.size(); i++) {
        Clause& c = ca[clauses[i]];
        bool unsat = true;
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) != l_False) {
                unsat = false;
                break;
            }
        }
        if (unsat) {
            return true;
        }
    }
    return false;
}

void testCheckConflict(int* clauses, unsigned* ends, unsigned* crefs, unsigned clauseCount, uint8_t* assigns, unsigned* conflict) {
    std::srand(9);
    for (unsigned idx = 0; idx < clauseCount; idx++) {
        unsigned startIdx = (idx == 0) ? 0 : ends[idx-1];
        unsigned endIdx = ends[idx];
        unsigned valCount[4];
        for (unsigned i = 0; i < 4; i++) {
            valCount[i] = 0;
        }
        for (unsigned i = startIdx; i < endIdx; i++) {
            uint8_t value = VALUE(clauses[i], assigns);
            valCount[value]++;
        }
        if (valCount[LF] == endIdx - startIdx) {
            // Fount a conflicting clause (evaluates to 0)
            unsigned cr = crefs[idx];
            *conflict = cr;
            if (std::rand() & 3 == 0) return;
        }
    }
}
