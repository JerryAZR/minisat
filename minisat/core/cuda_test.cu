#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"

void checkCudaError(const char msg[]) {
#ifdef CHECK_CUDA_ERROR
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("Error message: %s\n", msg);
        exit(1);
    }
#endif
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

void Minisat::Solver::getUnitClauses() {
    printf("Expected Unit clauses: ");
    for (int i = 0; i < clauses.size(); i++) {
        Clause& c = ca[clauses[i]];
        bool unsat = true;
        int count = 0;
        Lit imply = lit_Undef;
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_Undef) {
                count++;
                imply = c[j];
            }
            if (value(c[j]) == l_True) {
                unsat = false;
            }
        }
        if (!unsat || (count != 1)) {
            continue;
        } else {
            printf(" %d(%d) ", clauses[i], toInt(imply));
            printf("(");
            for (int j = 0; j < c.size(); j++) {
                printf("%d ", toInt(value(c[j])));
            }
            printf(")");
        }
    }
    printf("\n");
}
