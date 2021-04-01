#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"
#include <vector>

// interface (CPU) functions
using namespace Minisat;

/*_________________________________________________________________________________________________
|
|  propagate : [void]  ->  [Clause*]
|  
|  Description:
|    Propagates all enqueued facts. If a conflict arises, the conflicting clause is returned,
|    otherwise CRef_Undef.
|  
|    Post-conditions:
|      * the propagation queue is empty, even if there was a conflict.
|________________________________________________________________________________________________@*/
void Solver::propagate(std::vector<CRef>& hostConflicts) {
    CRef    confl     = CREF_UNDEF;
    int     num_props = 0;

    confl = checkConflictCaller(num_props);

    propagations += num_props;
    simpDB_props -= num_props;

    hostConflicts.clear();
    if (confl != CREF_UNDEF) hostConflicts.push_back(confl);
}
CRef Solver::propagate() {
    CRef    confl     = CRef_Undef;
    int     num_props = 0;

    while (qhead < trail.size()){
        Lit            p   = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher>&  ws  = watches.lookup(p);
        int        i, j, end;
        num_props++;

        // for (i = 0; i < ws.size(); i++) {
        //     CRef     cr        = ws[i].cref;
        //     Clause& c = ca[cr];

        //     unsigned startIdx = 0;
        //     unsigned endIdx = c.size();
        //     bool unsat = true;
        //     for (j = startIdx; j < endIdx; j++) {
        //         Lit variable = c[j];
        //         if (value(variable) != l_False) {
        //             unsat = false;
        //             break;
        //         }
        //     }
        //     if (unsat) {
        //         confl = cr;
        //         break;
        //     }
        // }
        // if (confl != CRef_Undef) {
        //     i = j;
        //     break;
        // }
        for (i = j = 0, end = ws.size(); i < end; i++){
            // Try to avoid inspecting the clause:
            Lit blocker = ws[i].blocker;
            if (value(blocker) == l_True){
                ws[j++] = ws[i]; continue; }

            // Make sure the false literal is data[1]:
            CRef     cr        = ws[i].cref;
            Clause&  c         = ca[cr];
            Lit      false_lit = ~p;
            if (c[0] == false_lit) {
                c[0] = c[1], c[1] = false_lit;
            }
            assert(c[1] == false_lit);

            // If 0th watch is true, then clause is already satisfied.
            Lit     first = c[0];
            Watcher w     = Watcher(cr, first);
            assert(value(first) == VALUE(first.x, assigns.begin()));
            if (value(first) == l_True){
                ws[j++] = w; continue; }

            // Look for new watch:
            bool flag = false;
            for (int k = 2; k < c.size(); k++) {
                if (value(c[k]) != l_False){
                    c[1] = c[k]; c[k] = false_lit;
                    watches[~c[1]].push(w);
                    flag = true;
                    break;
                }
            }
            if (flag) continue;

            // Did not find watch -- clause is unit under assignment:
            ws[j++] = w;
            if (value(first) == l_False){
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                i++;
                while (i < end)
                    ws[j++] = ws[i++];
                break;
            }else
                uncheckedEnqueue(first, cr);
        }
        ws.shrink(i - j);
    }
    propagations += num_props;
    simpDB_props -= num_props;

    return confl;
}

CRef Solver::checkConflictCaller(int& num_props) {
    
    CRef confl = CREF_UNDEF;
    unsigned implCount;
    while (true) {
        cudaMemset(deviceConfl, 0xFF, sizeof(unsigned));
        cudaMemset(deviceImplCount, 0, sizeof(unsigned));
        cudaAssignmentUpdate();
        checkCudaError("Failed to copy assignment data.\n");

        const size_t blockSize = 32;
        size_t gridSize = (clauses.size() - 1) / blockSize + 1;
        checkConflict<<<gridSize, blockSize>>>(
            (int*)deviceClauseVec.data, deviceClauseEnd.data, deviceCRefs.data,
            deviceCRefs.size, deviceAssigns, deviceLocks,
            deviceConfl, deviceImplications, deviceImplSource, deviceImplCount
        );
        checkCudaError("Error while launching kernel.\n");
        
        cudaMemcpy(&confl, deviceConfl, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&implCount, deviceImplCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
        checkCudaError("Failed to copy data back.\n");
        cudaDeviceSynchronize();
        // getUnitClauses();

        if (implCount > 0) {
            num_props += implCount;
            // Update variable assignment on the host side
            cudaMemcpy(hostImplications, deviceImplications, sizeof(int) * implCount, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostImplSource, deviceImplSource, sizeof(unsigned) * implCount, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            // printf("Found unit clause: ");
            // for (unsigned i = 0; i < implCount; i++) {
            //     printf(" %d ", hostImplSource[i]);
            // }
            // printf("\n");
            for (unsigned i = 0; i < implCount; i++) {
                CRef cr = hostImplSource[i];
                Clause& c = ca[cr];
                uncheckedEnqueue(hostImplications[i], hostImplSource[i]);
                for (int k = 0; k < c.size(); k++) {
                    if (value(c[k]) == l_True) {
                        Lit tmp = c[k];
                        c[k] = c[0];
                        c[0] = tmp;
                        break;
                    }
                }
            }
        }
        if (implCount == 0 || confl != CREF_UNDEF) break;
    }

    return confl;
}
// Cuda device functions

__global__ void checkConflict(int* clauses, unsigned* ends, unsigned* crefs, unsigned clauseCount,
    uint8_t* assigns, int* lock, unsigned* conflict, int* implications, unsigned* implSource, unsigned* implCount) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= clauseCount) return;

    unsigned startIdx = (idx == 0) ? 0 : ends[idx-1];
    unsigned endIdx = ends[idx];
    unsigned valCount[4];
    int implied = LIT_UNDEF;
    for (unsigned i = 0; i < 4; i++) {
        valCount[i] = 0;
    }
    for (unsigned i = startIdx; i < endIdx; i++) {
        uint8_t value = VALUE(clauses[i], assigns);
        valCount[value]++;
        if (value >= LU) implied = clauses[i];
    }
    if (valCount[LF] == endIdx - startIdx - 1 && valCount[LT] == 0) {
        // Found a unit clause
        if (atomicExch(lock+VAR(implied), 1) == 0) {
            // Obtain the lock and set the value
            // assigns[VAR(implied)] = SIGN(implied);
            unsigned writeIdx = atomicAdd(implCount, 1);
            implications[writeIdx] = implied;
            implSource[writeIdx] = crefs[idx];
        } else if (VALUE(implied, assigns) == LF) {
            // Failed to obtain lock.
            // conflict
            valCount[LF] = endIdx - startIdx;
        }
    }
    if (valCount[LF] == endIdx - startIdx) {
        // Fount a conflicting clause (evaluates to 0)
        unsigned cr = crefs[idx];
        atomicCAS(conflict, CREF_UNDEF, cr);
    }
}
