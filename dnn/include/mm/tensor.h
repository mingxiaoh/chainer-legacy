#pragma once

#include <vector>
#include "mkldnn.hpp"

using namespace std;

enum data_type_t {
    UNKNOWN_TYPE = 0,
    FLOAT32,
    SINT32,
}

class Tensor {
public:
    // Allocate memory in constructor
    Tensor(int ndim, vector<int> &dims, type=FLOAT32);
    Tensor(void *buf, size_t len, int ndim, vector<int> &dims, type=FLOAT32);
    void *getbuf() { return buf; }
    size_t get_len() { return len; }

protect:
    void *buf;
    size_t len;
    data_type_t type;
    int ndim;
    vector<int> dims;

private:
    mkldnn::memory::format internal_fmt;
};
