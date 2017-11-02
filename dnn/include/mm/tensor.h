#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include "mkldnn.hpp"

using namespace std;

enum data_type_t {
    UNKNOWN_TYPE = 0,
    FLOAT32,
    SINT32,
};

class Tensor {
public:
    // Allocate memory in constructor
    Tensor() : buf_(nullptr), len_(0), type_(UNKNOWN_TYPE), ndim_(0) {}
    Tensor(int ndim, vector<int> &dims, data_type_t type=FLOAT32);
    Tensor(void *buf, size_t len, int ndim, vector<int> &dims, data_type_t type=FLOAT32);
    void *getbuf() { return buf_; }
    size_t get_len() { return len_; }

protected:
    void *buf_;
    size_t len_;
    data_type_t type_;
    int ndim_;
    vector<int> dims_;

    mkldnn::memory::format mm_fmt_;
};

#endif
