#include "minisat/core/deviceVector.h"
#include "minisat/core/cuda.cuh"
#include <stdio.h>

using namespace Minisat;

void cudaDeviceMemcpy(void* dest, void* src, unsigned size);

deviceVector::deviceVector(unsigned size, unsigned cap) : data(nullptr), size(size), cap(cap) {
    if (cap) cudaMalloc(&data, cap * sizeof(unsigned));
}

deviceVector::~deviceVector() {
    if (cap) cudaFree(data);
}

deviceVector::deviceVector(const deviceVector& other) {
    size = other.size;
    cap = other.cap;
    if (cap) cudaMalloc(&data, cap * sizeof(unsigned));
    if (size) cudaDeviceMemcpy(data, other.data, size * sizeof(unsigned));
}

deviceVector& deviceVector::operator=(const deviceVector& other) {
    if (data == other.data) return *this;
    if (cap) cudaFree(data);
    size = other.size;
    cap = other.cap;
    if (cap) cudaMalloc(&data, cap * sizeof(unsigned));
    if (size) cudaDeviceMemcpy(data, other.data, size * sizeof(unsigned));
    return *this;
}

// Initialize deviceVector with a given host array and size
void deviceVector::init(unsigned* hostData, unsigned sz) {
    if (cap < sz) {
        while (cap < sz) cap = cap << 1;
        if (data) cudaFree(data);
        cudaMalloc(&data, cap * sizeof(unsigned));
        checkCudaError("Failed initialization - memory allocation.\n");
    }
    size = sz;
    cudaMemcpy(data, hostData, size * sizeof(unsigned), cudaMemcpyHostToDevice);
    checkCudaError("Failed initialization - memory copy.\n");
}

// Push an array of data
void deviceVector::bulk_push(unsigned* hostData, unsigned sz) {
    unsigned newSize = sz + size;
    reserve(newSize);
    checkCudaError("Failed bulk push reserve.\n");
    cudaMemcpy(data+size, hostData, sz * sizeof(unsigned), cudaMemcpyHostToDevice);
    size = newSize;
    checkCudaError("Failed bulk push memcpy\n");
}

// Avoid using this if possible
void deviceVector::push(unsigned newData) {
    reserve(size+1);
    cudaMemcpy(data+size, &newData, sizeof(unsigned), cudaMemcpyHostToDevice);
    size++;
}

// Resize the vector
void deviceVector::resize(unsigned newSize) {
    reserve(newSize);
    size = newSize;
}

// Raise cap to at least newCap
// May allocate more memory, but does not change the size
void deviceVector::reserve(unsigned newCap) {
    if (cap <= newCap) {
        // Calculate new cap
        while (cap <= newCap) cap = cap << 1;
        // Allocate more memory
        unsigned* tmpPtr = data;
        cudaMalloc(&data, cap * sizeof(unsigned));
        checkCudaError("Failed reserve malloc\n");
        // Copy existing data
        if (size) {
            cudaDeviceMemcpy(data, tmpPtr, size * sizeof(unsigned));
            checkCudaError("Failed reserve memcpy.\n");
            cudaFree(tmpPtr);
        }
    }
}

void cudaDeviceMemcpy(void* dest, void* src, unsigned size) {
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
}
