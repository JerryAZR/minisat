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
    int     num_props = 0;

    checkConflictCaller(num_props, hostConflicts);

    propagations += num_props;
    simpDB_props -= num_props;
}

void Solver::checkConflictCaller(int& num_props, std::vector<CRef>& hostConflicts) {
    unsigned implCount;
    unsigned conflCount;
    hostConflicts.clear();
    cudaAssignmentUpdate();
    while (true) {
        cudaMemset(deviceConflCount, 0, sizeof(unsigned));
        cudaMemset(deviceImplCount, 0, sizeof(unsigned));
        checkCudaError("Failed to copy assignment data.\n");

        size_t gridSize = (deviceClauseEnd.size - 1) / BLOCK_SIZE + 1;
        checkConflict<<<gridSize, BLOCK_SIZE>>>(
            (int*)deviceClauseVec.data, deviceClauseEnd.data, deviceCRefs.data,
            deviceCRefs.size, deviceAssigns, deviceLocks,
            deviceConfls,deviceConflCount , deviceImplications, deviceImplSource, deviceImplCount
        );
        checkCudaError("Error while launching kernel.\n");
        
        cudaMemcpy(&conflCount, deviceConflCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&implCount, deviceImplCount, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        checkCudaError("Failed to copy metadata back.\n");

        if (implCount > 0) {
            num_props += implCount;
            // Update variable assignment on the host side
            cudaMemcpy(hostImplications, deviceImplications, sizeof(int) * implCount, cudaMemcpyDeviceToHost);
            cudaMemcpy(hostImplSource, deviceImplSource, sizeof(unsigned) * implCount, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            checkCudaError("Failed to copy propagation assignments back.\n");

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
        if (conflCount > 0) {
            if (conflCount > MAX_CONFL) { conflCount = MAX_CONFL; }
            hostConflicts.resize(conflCount);
            cudaMemcpy(hostConflicts.data(), deviceConfls, conflCount * sizeof(unsigned), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            checkCudaError("Failed to copy conflicting clauses back.\n");
        }
        if (implCount == 0 || conflCount > 0) break;
    }
}
// Cuda device functions

__global__ void checkConflict(int* clauses, unsigned* ends, unsigned* crefs, unsigned clauseCount, uint8_t* assigns,
    int* lock, unsigned* conflicts, unsigned* conflCount, int* implications, unsigned* implSource, unsigned* implCount) {
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
            assigns[VAR(implied)] = SIGN(implied);
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
        unsigned conflIdx = atomicAdd(conflCount, 1);
        if (conflIdx < MAX_CONFL) {
            conflicts[conflIdx] = cr;
        }
    }
}
