#ifndef Minisat_DeviceVector_h
#define Minisat_DeviceVector_h

namespace Minisat {

// deviceVector class for managing CUDA device memory
class deviceVector {
public:
    unsigned* data;
    unsigned cap;
    unsigned size;

    deviceVector(unsigned size = 0, unsigned cap = 1024);
    ~deviceVector();

    void init(unsigned* hostData, unsigned sz);
    void bulk_push(unsigned* hostData, unsigned sz);
    void push(unsigned newData);
    void resize(unsigned newSize);
    void reserve(unsigned newCap);
};

}

#endif
