#include "minisat/core/cuda.cuh"
#include "minisat/core/Solver.h"
#include <vector>

using namespace Minisat;

void Solver::hostVecInit() {
#ifdef USE_CUDA
    hostClauseVec.clear();
    hostClauseEnd.clear();
    for (int i = 0; i < clauses.size(); i++) {
        CRef cr = clauses[i];
        Clause& c = ca[cr];
        if (c.learnt()) continue;
        for (int j = 0; j < c.size(); j++) {
            hostClauseVec.push_back(c[j]);
        }
        if (c.size() == 50) {
            printf("Idx %d has size 50\n", i);
        }
        hostClauseEnd.push_back(hostClauseVec.size());
    }
#endif
}


void Solver::cudaClauseInit() {
#ifdef USE_CUDA
    size_t litCount = hostClauseVec.size();
    size_t clauseCount = hostClauseEnd.size();
    deviceClauseVec.init((unsigned*)hostClauseVec.data(), hostClauseVec.size());
    deviceClauseEnd.init((unsigned*)hostClauseEnd.data(), hostClauseEnd.size());
    deviceCRefs.init((unsigned*)clauses.data, clauses.size());
    checkCudaError("Failed to initialize memory for clause data.\n");
    cudaMalloc(&deviceConfl, sizeof(unsigned));
    cudaMalloc(&deviceAssigns, sizeof(uint8_t) * assigns.size());
    checkCudaError("Failed to allocate memory for assgnment data.\n");
#endif
}

void Solver::cudaClauseFree() {
#ifdef USE_CUDA
    cudaFree(deviceConfl);
    cudaFree(deviceAssigns);
    checkCudaError("Failed to free device memory.\n");
#endif
}

void Solver::cudaClauseUpdate() {
#ifdef USE_CUDA
    size_t clauseCount = clauses.size();
    cudaMemcpy(deviceCRefs.data, clauses.data, clauseCount * sizeof(unsigned), cudaMemcpyHostToDevice);
    checkCudaError("Failed to update.\n");
#endif
}

void Solver::cudaAssignmentUpdate() {
#ifdef USE_CUDA
    cudaMemcpy(deviceAssigns, assigns.begin(), sizeof(uint8_t) * assigns.size(), cudaMemcpyHostToDevice);
#endif
}