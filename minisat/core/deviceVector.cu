#include "minisat/core/deviceVector.h"

using namespace Minisat;

void cudaDeviceMemcpy(void* dest, void* src, unsigned size);

deviceVector::deviceVector(unsigned size, unsigned cap) : data(nullptr), size(size), cap(cap) {
    if (cap) {
        cudaMalloc(&data, cap * sizeof(unsigned));
    }
}

deviceVector::~deviceVector() {
    if (cap) {
        cudaFree(data);
    }
}

// Initialize deviceVector with a given host array and size
void deviceVector::init(unsigned* hostData, unsigned sz) {
    if (cap < sz) {
        while (cap < sz) cap = cap << 1;
        if (data) cudaFree(data);
        cudaMalloc(&data, cap * sizeof(unsigned));
    }
    size = sz;
    cudaMemcpy(data, hostData, size * sizeof(unsigned), cudaMemcpyHostToDevice);
}

// Push an array of data
void deviceVector::bulk_push(unsigned* hostData, unsigned sz) {
    unsigned newSize = sz + size;
    reserve(newSize);
    cudaMemcpy(data+size, hostData, sz * sizeof(unsigned), cudaMemcpyHostToDevice);
    size = newSize;
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
        // Copy existing data
        if (size) {
            cudaDeviceMemcpy(data, tmpPtr, size * sizeof(unsigned));
            cudaFree(tmpPtr);
        }
    }
}

void cudaDeviceMemcpy(void* dest, void* src, unsigned size) {
    cudaMemcpy(dest, src, size * sizeof(unsigned), cudaMemcpyDeviceToDevice);
}
