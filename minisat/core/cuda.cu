#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"
#include <vector>
#include <stdlib.h>

using std::vector;
// #define CUDATEST

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
CRef Solver::propagate()
{
    CRef    confl     = CRef_Undef;
    int     num_props = 0;

    while (qhead < trail.size()){
        Lit            p   = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher>&  ws  = watches.lookup(p);
        num_props++;

        int* hostEnds = (int*)malloc(ws.size() * sizeof(int));
        CRef* hostCrefs = (CRef*)malloc(ws.size() * sizeof(CRef));
        vector<int> hostClauses;
        int totalSize = 0;
        int clauseCount = 0;
        int watcherCount = 0;
        int diffSize = 0;
        // Determine the size of each clause
        for (int idx = 0; idx < ws.size(); idx++) {
            Lit blocker = ws[idx].blocker;
            if (value(blocker) == l_True) {
                ws[watcherCount++] = ws[idx];
                continue;
            }
            Clause& cl = ca[ws[idx].cref];
            Lit false_lit = ~p;
            if (cl[0] == false_lit) {
                cl[0] = cl[1];
                cl[1] = false_lit;
            }
            totalSize += cl.size();
            hostCrefs[clauseCount] = ws[idx].cref;
            hostEnds[clauseCount++] = totalSize;
            for (int litIdx = 0; litIdx < cl.size(); litIdx++) {
                hostClauses.push_back(toInt(cl[litIdx]));
            }
        }
        int* hostActions = (int*)malloc(clauseCount * sizeof(int));
        if (clauseCount) {
            int* deviceActions, * deviceClauses, * deviceEnds;
            uint8_t* deviceAssigns;
            int* numConflicts, * conflictIndices;
            cudaMalloc(&deviceActions, clauseCount * sizeof(int));
            cudaMalloc(&deviceClauses, totalSize * sizeof(int));
            cudaMalloc(&deviceEnds, clauseCount * sizeof(int));
            cudaMalloc(&deviceAssigns, assigns.size() * sizeof(uint8_t));
            cudaMalloc(&conflictIndices, clauseCount * sizeof(int));
            cudaMalloc(&numConflicts, sizeof(int));
            cudaMemset(numConflicts, 0, sizeof(int));

            cudaMemcpy(deviceEnds, hostEnds, clauseCount * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceClauses, hostClauses.data(), totalSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceAssigns, assigns.begin(), assigns.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

            const size_t blockSize = 32;
            size_t gridSize = ((totalSize-1) / blockSize) + 1;
            propagateKernel<<<gridSize, blockSize>>>(clauseCount, deviceClauses, deviceEnds, deviceAssigns, deviceActions);
            cudaDeviceSynchronize();

            cudaMemcpy(hostActions, deviceActions, clauseCount * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hostClauses.data(), deviceClauses, totalSize * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(deviceClauses);
            cudaFree(deviceAssigns);
            cudaFree(deviceEnds);
            cudaFree(deviceActions);
            cudaFree(numConflicts);
            cudaFree(conflictIndices);
        }
        
        bool flag = false;
        // Iterate over the returned clauses and update the data on CPU
        for (int idx = 0; idx < clauseCount; idx++) {
            int action = hostActions[idx];
            CRef cr = hostCrefs[idx];
            int beginIdx = (idx == 0) ? 0 : hostEnds[idx-1];
            int* c = hostClauses.data() + beginIdx;
            Lit first;
            first.x = c[0];
            Watcher w = Watcher(cr, first);
            switch (action)
            {
            case CL_NEWBLOCK: ws[watcherCount++] = w; break;
            case CL_NEWWATCH: 
                Lit second;
                second.x = c[1];
                watches[~second].push(w);
                assert(ca[cr].size() == (hostEnds[idx] - beginIdx));
                memcpy(&((ca[cr])[0]), c, ca[cr].size());
                diffSize++;
                break;
            case CL_UNIT: 
                if (value(first) == l_Undef) {
                    uncheckedEnqueue(first, cr);
                }
                ws[watcherCount++] = w;
                break;
            case CL_CONFLICT:
                ws[watcherCount++] = w;
                confl = cr;
                // qhead = trail.size();
                // // Copy the remaining watches:
                // while (watcherCount + diffSize < ws.size()) {
                //     ws[watcherCount] = ws[watcherCount + diffSize];
                //     watcherCount++; 
                // }
                // flag = true;
                break;
            default: break;
            }
            if (flag) break;
        }
        free(hostEnds);
        free(hostCrefs);
        free(hostActions);
        ws.shrink(diffSize);
    }
    propagations += num_props;
    simpDB_props -= num_props;

    return confl;
}
#else
CRef Solver::propagate()
{
    CRef    confl     = CRef_Undef;
    int     num_props = 0;

    while (qhead < trail.size()){
        Lit            p   = trail[qhead++];     // 'p' is enqueued fact to propagate.
        vec<Watcher>&  ws  = watches.lookup(p);
        Watcher        *i, *j, *end;
        num_props++;

        for (i = j = (Watcher*)ws, end = i + ws.size(); i != end; i++){
            // Try to avoid inspecting the clause:
            Lit blocker = i->blocker;
            if (value(blocker) == l_True){
                *j++ = *i; continue; }

            // Make sure the false literal is data[1]:
            CRef     cr        = i->cref;
            Clause&  c         = ca[cr];
            Lit*     rawC      = &c[0];
            assert(LT == toInt(l_True));
            assert(LF == toInt(l_False));
            for (int n = 0; n < c.size(); n++) {
                assert(rawC[n] == c[n]);
            }
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
                *j++ = w; continue; }

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
            *j++ = w;
            if (value(first) == l_False){
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                i++;
                while (i < end)
                    *j++ = *i++;
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

// Cuda device functions
__global__ void propagateKernel(int clauseCount, int* clauses, int* clausesEnd, uint8_t* assigns, int* actions) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < clauseCount) {
        int clauseStart = (idx == 0) ? 0 : clausesEnd[idx - 1];
        int* c = clauses + clauseStart;
        int c_size = clausesEnd[idx] - clauseStart;
        int first = c[0];
        if (VALUE(first, assigns) == LT) {
            // update blcker
            actions[idx] = CL_NEWBLOCK;
            return;
        }
        // Look for new watch (i.e unassigned variable):
        for (int k = 2; k < c_size; k++) {
            if (VALUE(c[k], assigns) != LF) {
                int temp = c[1];
                c[1] = c[k];
                c[k] = temp;
                // watch new literal
                actions[idx] = CL_NEWWATCH;
                return;
            }
        }
        if (VALUE(first, assigns) == LF) {
            // conflict
            actions[idx] = CL_CONFLICT;
        } else {
            // unit clause;
            actions[idx] = CL_UNIT;
        }
    }
}

/* checkWatcher
 * Possible outcomes:
 * - Clause already satisfied (blocker == true) -- do nothing
 * - Clause already satisfied (blocker != true) -- update current blocker
 * - Clause undetermined (At least 2 unassigned vars) -- watch another literal
 * - Clause unit -- unit propagation
 * - Clause unsatisfied -- keep the remaining clauses and terminate
 * Return true if watcher needs to be updated. false if not
 */
int checkWatcher(int blocker, int p, int* c, unsigned c_size, uint8_t* assigns) {
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
