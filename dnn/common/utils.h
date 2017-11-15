#ifndef _UTILS_H_
#define _UTILS_H_

#include <glog/logging.h>
#include <mkldnn.hpp>
#include <iostream>
#include "op_param.h"
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
//// int16_t -> memory::data_type::s16
//// int8_t -> memory::data_type::s8
//// uint8_t -> memory::data_type::u8
//
template<typename T>
static mkldnn::memory::data_type memory_data_type() {
    if (typeid(T) == typeid(float))
        return mkldnn::memory::data_type::f32;
    else if (typeid(T) == typeid(int))
        return mkldnn::memory::data_type::s32;
    else if (typeid(T) == typeid(int16_t))
        return mkldnn::memory::data_type::s16;
    else if (typeid(T) == typeid(int8_t))
        return mkldnn::memory::data_type::s8;
    else if (typeid(T) == typeid(uint8_t))
        return mkldnn::memory::data_type::u8;

    LOG(ERROR) << "Not support type";
    return mkldnn::memory::data_type::data_undef;
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

static inline std::string long_to_string(size_t value) {
    std::ostringstream os;
    os << std::hex << "L" << value << "_";
    return os.str();
}

static inline mkldnn::algorithm pooling_algo_convert(pooling_param_t::algorithm input) {
    switch(input) {
        case pooling_param_t::algorithm::pooling_max:
            return mkldnn::pooling_max;
        case pooling_param_t::algorithm::pooling_avg:
            return mkldnn::pooling_avg;
        case pooling_param_t::algorithm::pooling_avg_include_padding:
            return mkldnn::pooling_avg_include_padding;
        case pooling_param_t::algorithm::pooling_avg_exclude_padding:
            return mkldnn::pooling_avg_exclude_padding;
        default:
            LOG(ERROR) << "Not a valid pooling algo";
            return mkldnn::pooling_max;
    }
}

#endif // _UTILS_H_
