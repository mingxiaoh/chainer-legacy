#pragma once

#include <vector>
#include "mkldnn.hpp"

using namespace std;

enum data_type_t {
    UNKNOWN_TYPE = 0,
    FLOAT32,
    SINT32,
};

inline mkldnn::memory::format public_format(mkldnn::memory::format origin)
{
    mkldnn::memory::format ret;
    // review this relations carefully
    switch(origin) {
        case mkldnn::memory::nchw:
        case mkldnn::memory::nhwc:
        case mkldnn::memory::chwn:
        case mkldnn::memory::nChw8c:
        case mkldnn::memory::nChw16c:
            ret = mkldnn::memory::nchw;
            break;
        case mkldnn::memory::oihw:
        case mkldnn::memory::ihwo:
        case mkldnn::memory::hwio:
        case mkldnn::memory::OIhw8i8o:
        case mkldnn::memory::OIhw16i16o:
        case mkldnn::memory::OIhw8o8i:
        case mkldnn::memory::OIhw16o16i:
        case mkldnn::memory::OIhw8i16o2i:
        case mkldnn::memory::OIhw8o16i2o:
        case mkldnn::memory::Oihw8o:
        case mkldnn::memory::Oihw16o:
        case mkldnn::memory::Ohwi8o:
        case mkldnn::memory::Ohwi16o:
        case mkldnn::memory::OhIw16o4i:
            ret = mkldnn::memory::oihw;
            break;
        default:
            ret = origin;
            break;
    }

    return ret;
}

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
