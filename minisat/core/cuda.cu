#include "minisat/core/cuda.cuh"
// Cuda functions

/* check_watcher
 * Possible outcomes:
 * - Clause already satisfied (blocker == true) -- do nothing
 * - Clause already satisfied (blocker != true) -- update current blocker
 * - Clause undetermined (At least 2 unassigned vars) -- watch another literal
 * - Clause unit -- unit propagation
 * - Clause unsatisfied -- keep the remaining clauses and terminate
 * Return true if watcher needs to be updated. false if not
 */
int check_watcher(int blocker, int p, int* c, unsigned c_size, uint8_t* assigns) {
    if (VALUE(blocker, assigns) == LT) {
        return CL_NOCHANGE; // Clause Satisfied
    }
    
    int false_lit = p ^ 1;
    if (c[0] == false_lit) {
        c[0] = c[1];
        c[1] = false_lit;
    }
    assert(c[1] == false_lit);

    int first = c[0];
    if (VALUE(first, assigns) == LT) {
        // update blcker
        return CL_NEWBLOCK;
    }

    // Look for new watch (i.e unassigned variable):
    for (int k = 2; k < c_size; k++) {
        if (VALUE(c[k], assigns) != LF) {
            c[1] = c[k]; c[k] = false_lit;
            // watch new literal
            return CL_NEWWATCH;
        }
    }

    if (VALUE(first, assigns) == LF) {
        // conflict
        return CL_CONFLICT;
    } else {
        // unit clause;
        return CL_UNIT;
    }
}
