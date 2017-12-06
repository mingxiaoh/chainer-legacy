/*
 *COPYRIGHT
 *All modification made by Intel Corporation: © 2017 Intel Corporation.
 *Copyright (c) 2015 Preferred Infrastructure, Inc.
 *Copyright (c) 2015 Preferred Networks, Inc.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#include <mkldnn.hpp>
#include <vector>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include "tensor.h"
#include "sum.h"

using namespace mkldnn;

static inline bool optimized_format(Tensor *t) {
    switch(t->format()) {
    case mkldnn_nChw16c:
    case mkldnn_nChw8c:
    case mkldnn_OIhw8i8o:
    case mkldnn_OIhw16i16o:
    case mkldnn_OIhw8i16o2i:
    case mkldnn_OIhw8o16i2o:
    case mkldnn_OIhw8o8i:
    case mkldnn_OIhw16o16i:
    case mkldnn_Oihw8o:
    case mkldnn_Oihw16o:
        return true;
    default:
        return false;
    }
}

template<typename T>
static T * sum_nChwXC_along_channel(T *src, mkldnn_memory_format_t format,
                                    mkldnn_dims_t dims, vector<int> axis, T *dst) {
    int mb = dims[0],
        ic = dims[1],
        ih = dims[2],
        iw = dims[3],
        cg = format == mkldnn_nChw16c ?
             16 : 8,
        cn = ic / cg;

    int blk_nthr = omp_get_max_threads(),
        blk_num = blk_nthr,
        blk_len = mb / blk_num,
        blk_len_ex = mb % blk_num;

    if (!blk_len)
        blk_nthr = mb;

    T *buf = reinterpret_cast<T *>(new avx::byte[ic * blk_nthr * sizeof(T)]);

    # pragma omp parallel num_threads(blk_nthr)
    {
        int ithr = omp_get_thread_num();
        int blen = ithr < blk_len_ex ? blk_len + 1 : blk_len;
        int bstart = ithr <= blk_len_ex ? (blk_len + 1) * ithr :
                     blk_len_ex * (blk_len + 1) + (ithr - blk_len_ex) * blk_len;
        int bend = bstart + blen;

        T *loc_src = src + bstart * ic * ih * iw;
        for (int b = bstart; b < bend; b++) {
            T *loc_buf = buf + ithr * ic;
            for (int c = 0; c < cn; c++) {
                if (b == bstart)
                    for (int o = 0; o < cg; o++)
                        loc_buf[o] = 0;

                for (int hw = 0; hw < ih * iw; hw++) {
                    for (int o = 0; o < cg; o++)
                        loc_buf[o] += loc_src[o];
                    loc_src += cg;
                }

                loc_buf += cg;
            }
        }
    }

    // Allreduce
    int c_nthr = omp_get_max_threads(),
        c_num = c_nthr,
        c_len = ic / c_num,
        c_len_ex = ic % c_num;

    if (!c_len)
        c_nthr = ic;

    # pragma omp parallel num_threads(c_nthr)
    {
        int ithr = omp_get_thread_num();
        int clen = ithr < c_len_ex ? c_len + 1 : c_len;
        int cstart = ithr <= c_len_ex ? (c_len + 1) * ithr :
                     c_len_ex * (c_len + 1) + (ithr - c_len_ex) * c_len;
        int cend = cstart + clen;

        for (int c = cstart; c < cend; c++)
            dst[c] = 0;

        for (int i = 0; i < blk_nthr; i++) {
            T *loc_buf = buf + i * ic;
            for (int c = cstart; c < cend; c++)
                dst[c] += loc_buf[c];
        }
    }

    delete(reinterpret_cast<avx::byte *>(buf));

    return dst;
}

// 4 dimensions(NCHW/OIHW) opitimzation for mkldnn backend only.
Tensor * sum_opt_along_axis(Tensor *src, vector<int> axis) {
    int axises = axis.size();
    vector<int> valid_axis_4dim = {0, 2, 3};

    if (src->ndims() != 4 || axises != 3) {
        return nullptr;
    }

    auto valid_axis = [](int axises,
                         vector<int> axis,
                         vector<int> valid_axis) -> bool {
        for (int i = 0; i < axises; i++) {
            if (valid_axis[i] != axis[i])
                return false;
        }
        return true;
    };

    try {
        switch (src->format()) {
        case mkldnn_nChw8c:
            if (!valid_axis(axises, axis, valid_axis_4dim))
                throw std::runtime_error(
                    "Invalid axis in tensor sum along axis <mkldnn_nChw8c>");
            break;
        case mkldnn_nChw16c:
            if (!valid_axis(axises, axis, valid_axis_4dim))
                throw std::runtime_error(
                    "Invalid axis in tensor sum along axis <mkldnn_nChw16c>");
            break;
        default:
            throw std::runtime_error(
                "Invalid format in tensor sum along axis");
            break;
        }
    } catch (std::runtime_error &e) {
        (void)e;
        return nullptr;
    }

    Tensor *dst = nullptr;
    try {
        switch (src->type()) {
        case FLOAT32:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<float *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<float *>(dst->data()));
            break;
        case SINT32:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int32_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int32_t *>(dst->data()));
            break;
        case SINT16:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int16_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int16_t *>(dst->data()));
            break;
        case SINT8:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<int8_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<int8_t *>(dst->data()));
            break;
        case UINT8:
            dst = new Tensor(1, {src->desc().data.dims[1]}, src->type());
            sum_nChwXC_along_channel(static_cast<uint8_t *>(src->data()), src->format(),
                    src->desc().data.dims, axis, static_cast<uint8_t *>(dst->data()));
            break;
        default:
            throw std::runtime_error("Invalid dtype in tensor sum along axis");
            break;
        }
    } catch (std::runtime_error &e) {
        (void)e;
        return nullptr;
    }

    return dst;
}

Tensor * sum_common_along_axis(Tensor *src, vector<int> axis) {
    return nullptr;
}

Tensor * sum_along_axis(Tensor *src, vector<int> axis) {
    if (optimized_format(src))
        return sum_opt_along_axis(src, axis);
    else
        return sum_common_along_axis(src, axis);
}
