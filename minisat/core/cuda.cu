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

        if (implCount > 0) {
            num_props += implCount;
            // Update variable assignment on the host side
            cudaMemcpy(hostImplications, deviceImplications, sizeof(int) * implCount, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostImplSource, deviceImplSource, sizeof(unsigned) * implCount, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

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
