#ifndef Minisat_DeviceVector_h
#define Minisat_DeviceVector_h

namespace Minisat {

// deviceVector class for managing CUDA device memory
// Not using thrust/device_vector because
//  this file needs to be included in other cpp (*.cc) files
class deviceVector {
public:
    unsigned* data;
    unsigned cap;
    unsigned size;

    deviceVector(unsigned size = 0, unsigned cap = 1024);
    // I don't think I'll need the copy constructor and copy assignment operator
    // but might as well define them by the rule of 3.
    deviceVector(const deviceVector& other);
    ~deviceVector();
    deviceVector& operator=(const deviceVector& other);

    void init(unsigned* hostData, unsigned sz);
    void bulk_push(unsigned* hostData, unsigned sz);
    void push(unsigned newData);
    void resize(unsigned newSize);
    void reserve(unsigned newCap);
};

}

#endif
