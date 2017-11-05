#pragma once

#include <vector>
#include "mkldnn.hpp"

using namespace std;
using namespace mkldnn;
extern engine cpu_engine;

typedef size_t size_type;
enum data_type_t {
    UNKNOWN_TYPE = 0,
    FLOAT32,
    SINT32,
};

inline int type2size(data_type_t type) {
    int size = 0;
    switch (type) {
        case FLOAT32:
            size = 4;
            break;
        case SINT32:
            size = 4;
            break;
        default:
            break;
    }
    return size;
}

inline size_t prod(vector<int>dims, int ndims)
{
    size_t prod = 1;
    for (int i = (ndims - 1); i >= 0; i--) {
        prod *= dims[i];
    }
    return prod;
}

inline mkldnn_memory_format_t ndims2format(int ndims)
{
    mkldnn_memory_format_t fmt = mkldnn_any; 
    switch (ndims) {
        case 1:
            fmt = mkldnn_x;
            break;
        case 2:
            fmt = mkldnn_nc;
            break;
        case 4:
            fmt = mkldnn_nchw;
            break;
        default:
            throw mkldnn::error(mkldnn_invalid_arguments
                    , "MKLDNN does not support dimensions"
                    + ndims);
    }

    return fmt;
}

inline mkldnn_memory_format_t public_format(mkldnn_memory_format_t origin)
{
    mkldnn_memory_format_t ret;
    // review this relations carefully
    switch(origin) {
        case mkldnn_nchw:
        case mkldnn_nhwc:
        case mkldnn_chwn:
        case mkldnn_nChw8c:
        case mkldnn_nChw16c:
            ret = mkldnn_nchw;
            break;
        case mkldnn_oihw:
        case mkldnn_ihwo:
        case mkldnn_hwio:
        case mkldnn_OIhw8i8o:
        case mkldnn_OIhw16i16o:
        case mkldnn_OIhw8o8i:
        case mkldnn_OIhw16o16i:
        case mkldnn_OIhw8i16o2i:
        case mkldnn_OIhw8o16i2o:
        case mkldnn_Oihw8o:
        case mkldnn_Oihw16o:
        case mkldnn_Ohwi8o:
        case mkldnn_Ohwi16o:
        case mkldnn_OhIw16o4i:
            ret = mkldnn_oihw;
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
    Tensor() : ndims_(0), type_(UNKNOWN_TYPE), size_(0), data_(nullptr) {}

    Tensor(int ndims, vector<int> &dims, data_type_t type=FLOAT32)
        : ndims_(ndims), dims_(dims), type_(type) {
            size_ = std::accumulate(dims.begin(), dims.begin() + ndims, 1
                    , std::multiplies<int>());
            data_ = std::shared_ptr<avx::byte>(new avx::byte [len()]
                    , [] (avx::byte *p) {delete [] p;});
            mm_fmt_ = ndims2format(ndims);
            memory::data_type dt = to_mkldnn_type();
            mem_.reset(new mkldnn::memory(
                        { { { dims_ }, dt, static_cast<memory::format>(mm_fmt_) }
                        , cpu_engine }, data_.get()));
        }

    Tensor(int ndims, vector<int> &dims, void *buf, data_type_t type=FLOAT32)
        : ndims_(ndims), dims_(dims), type_(type) {
            size_ = std::accumulate(dims.begin(), dims.begin() + ndims, 1
                    , std::multiplies<int>());
            data_ = std::shared_ptr<avx::byte>(new avx::byte [len()]
                    , [] (avx::byte *p) {delete [] p;});
            memcpy(data_.get(), buf, len());
            mm_fmt_ = ndims2format(ndims);
            memory::data_type dt = to_mkldnn_type();
            mem_.reset(new mkldnn::memory(
                        { { { dims_ }, dt, static_cast<memory::format>(mm_fmt_) }
                        , cpu_engine }, data_.get()));
        }

    Tensor(mkldnn::memory::dims &dims
        , mkldnn::memory::data_type dt
        , mkldnn::memory::format format
        , const mkldnn::engine &engine)
            : Tensor({{std::move(dims), dt, format}, engine}) {}

    Tensor(mkldnn::memory::primitive_desc pd) {
        auto md = pd.desc().data;
        ndims_ = md.ndims;
        dims_.assign(md.dims, md.dims + md.ndims);
        type_ = to_tensor_type(md.data_type);
        size_ = std::accumulate(md.dims, md.dims + md.ndims, 1
                , std::multiplies<int>());
        data_ = std::shared_ptr<avx::byte>(new avx::byte [len()]
                , [] (avx::byte *p) {delete [] p;});
        mm_fmt_ = md.format;
        memory::data_type dt = to_mkldnn_type();
        mem_.reset(new mkldnn::memory(
                    { { { dims_ }, dt , static_cast<memory::format>(mm_fmt_) }
                    , cpu_engine }, data_.get()));
    }

    inline size_t len() {
        return size_ * type2size(type_);
    }

    inline bool incompatible() const {
        return (public_format(mm_fmt_) == mm_fmt_);
    }

    inline memory::data_type to_mkldnn_type() const {
        memory::data_type type;
        switch (type_) {
            case FLOAT32:
                type = memory::data_type::f32;
                break;
            case SINT32:
                type = memory::data_type::s32;
                break;
            default:
                type = memory::data_undef;
                break;
        }
        return type;
    }

    inline data_type_t to_tensor_type(mkldnn_data_type_t type) const {
        data_type_t dt;
        switch (type) {
            case mkldnn_f32:
                dt = FLOAT32;
                break;
            case mkldnn_s32:
                dt = SINT32;
                break;
            default:
                dt = UNKNOWN_TYPE;
                break;
        }
        return dt;
    }

    inline void *data() const { return data_.get(); }
    inline size_type size() const { return size_; }
    inline mkldnn::engine get_engine() const {
        return cpu_engine;
    }

    inline int ndims() const {
        return ndims_;
    }

    inline shared_ptr<mkldnn::memory> to_mkldnn_memory() const {
        return mem_;
    }

    inline memory::desc desc() const {
        return mem_->get_primitive_desc().desc();
    }

    inline mkldnn_memory_format_t format() const {
        return mm_fmt_;
    }

    inline mkldnn::memory::format cxx_format() const {
        return static_cast<mkldnn::memory::format>(mm_fmt_);
    }

    inline mkldnn::memory::dims cxx_dims() const {
        mkldnn::memory::dims ret(dims_.begin(), dims_.begin() + ndims_);
        return ret;
    }

    inline mkldnn::memory::data_type cxx_data_type() const {
        return static_cast<mkldnn::memory::data_type>(to_mkldnn_type());
    }

protected:
    int ndims_;
    vector<int> dims_;
    data_type_t type_;
    size_t size_;
    size_t len_;
    std::shared_ptr<avx::byte> data_;

    mkldnn_memory_format_t mm_fmt_;
    std::shared_ptr<mkldnn::memory> mem_;
private:
};
