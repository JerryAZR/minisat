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
        c.sendToGPU();
        hostClauseEnd.push_back(hostClauseVec.size());
    }
#endif
}


void Solver::cudaClauseInit() {
#ifdef USE_CUDA
    size_t varCount = assigns.size();
    deviceClauseVec.init((unsigned*)hostClauseVec.data(), hostClauseVec.size());
    deviceClauseEnd.init((unsigned*)hostClauseEnd.data(), hostClauseEnd.size());
    deviceCRefs.init((unsigned*)clauses.data, clauses.size());
    checkCudaError("Failed to initialize memory for clause data.\n");
    cudaMalloc(&deviceConflCount, sizeof(unsigned));
    cudaMalloc(&deviceConfls, sizeof(unsigned) * MAX_CONFL);
    cudaMalloc(&deviceAssigns, sizeof(uint8_t) * varCount);
    cudaMalloc(&deviceImplCount, sizeof(unsigned));
    cudaMalloc(&deviceImplications, sizeof(int) * varCount);
    cudaMalloc(&deviceImplSource, sizeof(unsigned) * varCount);
    cudaMalloc(&deviceLocks, sizeof(int) * varCount);
    checkCudaError("Failed to allocate memory for assgnment data.\n");
    hostImplications = (Lit*)malloc(sizeof(int) * varCount);
    hostImplSource = (CRef*)malloc(sizeof(unsigned) * varCount);
#endif
}

void Solver::cudaClauseFree() {
#ifdef USE_CUDA
    cudaFree(deviceConflCount);
    cudaFree(deviceConfls);
    cudaFree(deviceAssigns);
    cudaFree(deviceImplCount);
    cudaFree(deviceImplications);
    cudaFree(deviceImplSource);
    cudaFree(deviceLocks);
    checkCudaError("Failed to free device memory.\n");
    free(hostImplications);
    free(hostImplSource);
#endif
}

void Solver::cudaClauseUpdate() {
#ifdef USE_CUDA
    static unsigned updateCount = 0;
    updateCount++;
    unsigned originalClauseCount = clauses.size();
    unsigned originalLitCount = hostClauseVec.size();
    // Update CRefs of original clauses
    cudaMemcpy(deviceCRefs.data, clauses.data, originalClauseCount * sizeof(unsigned), cudaMemcpyHostToDevice);
    // Update learnt clauses
    cudaLearntUpdate();
    checkCudaError("Failed to update.\n");
#endif
}

void Solver::cudaLearntUpdate() {
    unsigned originalClauseCount = clauses.size();
    unsigned originalLitCount = hostClauseVec.size();
    // Clear learnt clauses
    hostLearntVec.clear();
    hostLearntEnd.clear();
    deviceClauseVec.resize(originalLitCount);
    deviceClauseEnd.resize(originalClauseCount);
    deviceCRefs.resize(originalClauseCount);
    // Add learnt clauses to host vector
    for (int i = 0; i < learnts.size(); i++) {
        CRef cr = learnts[i];
        Clause& c = ca[cr];
        for (int j = 0; j < c.size(); j++) {
            hostLearntVec.push_back(c[j]);
        }
        hostLearntEnd.push_back(originalLitCount + hostLearntVec.size());
        c.sendToGPU();
    }
    // Copy host vector to device vector
    deviceClauseVec.bulk_push((unsigned*)hostLearntVec.data(), hostLearntVec.size());
    deviceClauseEnd.bulk_push((unsigned*)hostLearntEnd.data(), learnts.size());
    deviceCRefs.bulk_push((unsigned*)learnts.data, learnts.size());
}

void Solver::cudaLearntAppend(std::vector<CRef>& newLearnts) {
    hostLearntVec.clear();
    hostLearntEnd.clear();
    unsigned originalLitCount = deviceClauseVec.size;
    for (unsigned i = 0; i < newLearnts.size(); i++) {
        CRef cr = newLearnts[i];
        if (cr == CRef_Undef) continue;
        Clause& c = ca[cr];
        for (int j = 0; j < c.size(); j++) {
            hostLearntVec.push_back(c[j]);
        }
        hostLearntEnd.push_back(originalLitCount + hostLearntVec.size());
        c.sendToGPU();
    }
    deviceClauseVec.bulk_push((unsigned*)hostLearntVec.data(), hostLearntVec.size());
    deviceClauseEnd.bulk_push((unsigned*)hostLearntEnd.data(), hostLearntEnd.size());
    deviceCRefs.bulk_push((unsigned*)newLearnts.data(), newLearnts.size());
}

void Solver::cudaAssignmentUpdate() {
#ifdef USE_CUDA
    cudaMemcpy(deviceAssigns, assigns.begin(), sizeof(uint8_t) * assigns.size(), cudaMemcpyHostToDevice);
    cudaMemset(deviceLocks, 0, sizeof(int) * assigns.size());
#endif
}
