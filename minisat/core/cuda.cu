#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"
#include <stdlib.h>
#include <algorithm>
#include <vector>

#define CUDATEST

void checkCudaError(const char msg[]) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("Error message: %s\n", msg);
        exit(1);
    }
}

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
#ifdef CUDATEST
CRef Solver::propagate() {
    CRef    confl     = CREF_UNDEF;
    int     num_props = 0;

    while (qhead < trail.size()){
        Lit            p   = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher>&  ws  = watches.lookup(p);
        int        i, j, end;
        num_props++;
        
        // First check for conflicts
        bool run_cuda = (ws.size() > 64);
        if (run_cuda) {
            cudaMemset(deviceConfl, 0xFF, sizeof(unsigned));
            cudaMemcpy(deviceAssigns,
                assigns.begin(),
                sizeof(uint8_t) * assigns.size(),
                cudaMemcpyHostToDevice);
            checkCudaError("Failed to copy assignment data.\n");

            const size_t blockSize = 32;
            size_t gridSize = (clauses.size() - 1) / blockSize + 1;
            checkConflict<<<gridSize, blockSize>>>(
                deviceClauseVec, deviceClauseEnd, deviceCRefs,
                clauses.size(), deviceAssigns, deviceConfl
            );
            cudaDeviceSynchronize();
            checkCudaError("Error while launching kernel.\n");
            
            cudaMemcpy(&confl, deviceConfl, sizeof(unsigned), cudaMemcpyDeviceToHost);
            checkCudaError("Failed to copy data back.\n");
            verifyUnsat(confl);
        } else {
            for (i = 0; i < ws.size(); i++) {
                CRef cr = ws[i].cref;
                Clause& c = ca[cr];
                unsigned startIdx = 0;
                unsigned endIdx = c.size();
                bool unsat = true;
                for (j = startIdx; j < endIdx; j++) {
                    Lit variable = c[j];
                    if (value(variable) != l_False) {
                        unsat = false;
                        break;
                    }
                }
                if (unsat) {
                    confl = cr;
                    break;
                }
            }
        }
        
        if (confl == CREF_UNDEF) {
            confl = CRef_Undef;
            std::vector<Lit> tmpLits;
            std::vector<CRef> tmpCRefs;
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
                }else {
                    tmpCRefs.push_back(cr);
                    tmpLits.push_back(first);
                }
            }
            for (unsigned n = 0; n < tmpLits.size(); n++) {
                enqueue(tmpLits[n], tmpCRefs[n]);
            }
        } else {
            qhead = trail.size();
            i = j;
            break;
        }
        ws.shrink(i - j);
    }
    propagations += num_props;
    simpDB_props -= num_props;

    return confl;
}
#else
CRef Solver::propagate() {
    CRef    confl     = CRef_Undef;
    int     num_props = 0;

    while (qhead < trail.size()){
        Lit            p   = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher>&  ws  = watches.lookup(p);
        int        i, j, end;
        num_props++;

        for (i = 0; i < ws.size(); i++) {
            CRef     cr        = ws[i].cref;
            Clause& c = ca[cr];

            unsigned startIdx = 0;
            unsigned endIdx = c.size();
            bool unsat = true;
            for (j = startIdx; j < endIdx; j++) {
                Lit variable = c[j];
                if (value(variable) != l_False) {
                    unsat = false;
                    break;
                }
            }
            if (unsat) {
                confl = cr;
                break;
            }
        }
        if (confl != CRef_Undef) {
            i = j;
            break;
        }
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
#endif

void Solver::verifyUnsat(CRef cr) {
    if (cr == CREF_UNDEF) return;
    Clause& c = ca[cr];
    for (int i = 0; i < c.size(); i++) {
        if (value(c[i]) != l_False) {
            printf("False unsat.\n");
            exit(1);
        }
    }
}

void Solver::cudaClauseInit() {
#ifdef CUDATEST
    size_t litCount = hostClauseVec.size();
    size_t clauseCount = hostClauseEnd.size();
    cudaMalloc(&deviceClauseVec, litCount * sizeof(int));
    cudaMalloc(&deviceClauseEnd, clauseCount * sizeof(unsigned));
    cudaMalloc(&deviceCRefs, clauseCount * sizeof(unsigned));
    checkCudaError("Failed to allocate memory for clause data.\n");
    cudaMalloc(&deviceConfl, sizeof(unsigned));
    cudaMalloc(&deviceAssigns, sizeof(uint8_t) * assigns.size());
    checkCudaError("Failed to allocate memory for assgnment data.\n");

    cudaMemcpy(deviceClauseVec, hostClauseVec.data(), litCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClauseEnd, hostClauseEnd.data(), clauseCount * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCRefs, clauses.data, clauseCount * sizeof(unsigned), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy clause data to device memory.\n");
#endif
}

void Solver::cudaClauseFree() {
#ifdef CUDATEST
    cudaFree(deviceClauseEnd);
    cudaFree(deviceClauseVec);
    cudaFree(deviceCRefs);
    cudaFree(deviceConfl);
    cudaFree(deviceAssigns);
    checkCudaError("Failed to free device memory.\n");
#endif
}

void Solver::cudaClauseUpdate() {
#ifdef CUDATEST
    size_t clauseCount = clauses.size();
    cudaMemcpy(deviceCRefs, clauses.data, clauseCount * sizeof(unsigned), cudaMemcpyHostToDevice);
    checkCudaError("Failed to update.\n");
#endif
}

// Cuda device functions

__global__ void checkConflict(int* clauses, unsigned* ends, unsigned* crefs, unsigned clauseCount, uint8_t* assigns, unsigned* conflict) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= clauseCount) return;

    unsigned startIdx = (idx == 0) ? 0 : ends[idx-1];
    unsigned endIdx = ends[idx];
    int valCount[4];
    for (int i = 0; i < 4; i++) {
        valCount[i] = 0;
    }
    for (int i = startIdx; i < endIdx; i++) {
        uint8_t value = VALUE(clauses[i], assigns);
        valCount[value]++;
    }
    if (valCount[LF] == endIdx - startIdx) {
        // Fount a conflicting clause (evaluates to 0)
        unsigned cr = crefs[idx];
        atomicCAS(conflict, CREF_UNDEF, cr);
    }
}
