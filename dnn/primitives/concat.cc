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


#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "tensor.h"
#include "mem.h"
#include "concat.h"
#include "utils.h"
#include "concat_fwd.h"
#include "prim_factory.h"
#include "reorder_op.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Concat<T>::Concat()
{
}

template<typename T>
Concat<T>::~Concat()
{
}

template<typename T>
Tensor *Concat<T>::Forward(
                std::vector<Tensor*> src,
                int axis)
{
    // sanity check
    assert (axis == 1); // currently, only support concat as channel
    assert (src.size() > 0);

    std::vector<mkldnn::memory::format> src_fmts;
    std::vector<mkldnn::memory::format> expected_fmts;
    std::vector<void*> src_datas;
    std::vector<void*> src_reorder;

    std::vector<mkldnn::memory::dims> src_ds;
    mkldnn::memory::dims dst_d;

    //get output channel
    int out_channel = 0;
    for (int i = 0; i < src.size(); i++) {
        //get relate infor from src
        src_fmts.push_back(src[i]->cxx_format());
        src_datas.push_back(src[i]->data());
        src_reorder.push_back(src[i]->data());

        src_ds.push_back(src[i]->cxx_dims());
        out_channel += (src[i]->cxx_dims())[1];
    }
    dst_d = {src_ds[0][0], out_channel, src_ds[0][2], src_ds[0][3]};
    
    // get a concat fwd from primitive pool
    ConcatFwd<T> *concat_forward = NULL;
    concat_forward = ConcatFwdFactory<T>::get(src_ds, dst_d, axis);

    // check wehther fmt is same
    expected_fmts = concat_forward->src_fmts_;
    assert(src_fmts.size() == expected_fmts.size());

    for (int i = 0; i < expected_fmts.size(); i++) {
        if ( src_fmts[i] != expected_fmts[i]) {
            LOG(INFO) << "Concat src fmt not match ("<< i << "):"
                "src_fmt=" << src_fmts[i] <<
                "; expected_fmt="<< expected_fmts[i];
            // From reorder factory to find one reorder
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_ds[i], src_fmts[i], expected_fmts[i]);
            src_reorder[i] = new avx::byte[src[i]->len()];
            reorder_src_op->execute(src_datas[i], src_reorder[i]);
        }
    }

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    Tensor *dst_tensor = new Tensor(dst_d, src[0]->cxx_data_type(), concat_forward->dst_fmt_, cpu_engine);
    
    // do forward
    concat_forward->execute(src_reorder, dst_tensor->data());

    //FIXME here may cause performance issue
    for (int i = 0; i < src_reorder.size(); i++) {
        if (src_reorder[i] != src_datas[i]) {
            // means reorder happen
            delete static_cast<avx::byte *>(src_reorder[i]);
        }
    }

    return dst_tensor;
}


template class Concat<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
