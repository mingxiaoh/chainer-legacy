#ifndef _UTILS_H_
#define _UTILS_H_

#include <glog/logging.h>
#include <mkldnn.hpp>
#include <iostream>
using namespace mkldnn;

memory::format get_desired_format(int channel);

template<typename T>
void eltwise_multiply(T* x1, T* x2, T* y, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        y[i] = x1[i] * x2[i];
    }
}

//
//// map C type with mkldnn's
//// float -> memory::data_type::f32
//// int -> memory::data_type::s32
//
template<typename T>
static mkldnn::memory::data_type memory_data_type() {
    if (typeid(T) == typeid(float))
        return mkldnn::memory::data_type::f32;
    else if (typeid(T) == typeid(int))
        return mkldnn::memory::data_type::s32;

    LOG(ERROR) << "Not support type";
    return mkldnn::memory::data_type::data_undef;
}

inline void reorder_func (mkldnn::memory src, mkldnn::memory dst) {
    if ( src.get_primitive_desc() != dst.get_primitive_desc()) {
        std::shared_ptr<mkldnn::stream> reorder_stream;
        reorder_stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
        mkldnn::reorder reorder_prim = reorder(src, dst);
        reorder_stream->submit({reorder_prim});
    }
}

// utils function conver int/double/bool/dims/ to string
static std::string int_to_string(int value) {
    std::ostringstream os;
    os << std::hex << "I" << value << "_";
    return os.str();
}

static std::string double_to_string(double value) {
    std::ostringstream os;
    os << "D" << value << "_";
    return os.str();
}

static std::string bool_to_string(bool value) {
    std::ostringstream os;
    os << "D" << value << "_";
    return os.str();
}

static std::string dims_to_string(mkldnn::memory::dims dims) {
   std::ostringstream os;
   os << "DIMS:";
   for (int i = 0; i < dims.size(); i++)
       os << dims[i] << ",";
   os << ";";
   return os.str();
}

#endif // _UTILS_H_
