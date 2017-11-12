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
 *
 *######################################################################
 *# The CuPy is designed based on NumPy's API.
 *# CuPy's source code and documents contain the original NumPy ones.
 *######################################################################
 *Copyright (c) 2005-2016, NumPy Developers.
 *All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are
 *met:
 *
 *    * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    * Neither the name of the NumPy Developers nor the names of any
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *######################################################################
 */


#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "tensor.h"
#include "mem.h"
#include "pooling.h"
#include "utils.h"
#include "pooling_fwd.h"
#include "pooling_fwd_factory.h"
#include "reorder_op.h"
#include "reorder_factory.h"

using namespace mkldnn;

const mkldnn::memory::dims NONE_DIMS = {}; 
extern engine cpu_engine;

template<typename T>
Pooling2D<T>::Pooling2D()
{
}

template<typename T>
Pooling2D<T>::~Pooling2D()
{
}

template<typename T>
std::vector<Tensor *> Pooling2D<T>::Forward(
                Tensor *src,
                pooling_param_t *pp)
{
    std::vector<Tensor *> outputs;

    // sanity check
    mkldnn::memory::dims src_dims = {pp->src_d1, pp->src_d2, pp->src_d3, pp->src_d4};
    mkldnn::memory::dims dst_dims = {pp->dst_d1, pp->dst_d2, pp->dst_d3, pp->dst_d4};
    assert(src_dims == src->cxx_dims());

    //sanity check for data type
    //assuem all should have same data type as T
    //FIXME
    //yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == src->cxx_data_type());
    
    // get a conv2d fwd from primitive pool
    Pooling2DFwd<T> *pooling2d_forward = NULL;
    pooling2d_forward = Pooling2DFwdFactory<T>::get(src_dims, dst_dims,
                pp->kh, pp->kw,
                pp->sy, pp->sx, 
                pp->pad_lh, pp->pad_lw, pp->pad_rh, pp->pad_rw,
                pooling_algo_convert(pp->algo_kind));
    
    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor

    void *src_tmp = src->data();
    void *src_reorder = NULL;
    
    // check wehther fmt is same
    if (src_fmt == pooling2d_forward->src_fmt_) {
        LOG(INFO) << "pooling forward fmt matched";
    } else {
        LOG(INFO) << "pooling fwd fmt not match, need to reorder";

        if (src_fmt != pooling2d_forward->src_fmt_) {
            LOG(INFO) << "src_fmt=" << src_fmt <<", pooling2d_forward->src_fmt_=" << pooling2d_forward->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, pooling2d_forward->src_fmt_);
            src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder);
            src_tmp = src_reorder;
        }
    }

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    //Tensor *ws_tensor = new Tensor(dst_dims, src->cxx_data_type(), pooling2d_forward->ws_fmt_, cpu_engine);
    Tensor *dst_tensor = new Tensor(dst_dims, src->cxx_data_type(), pooling2d_forward->dst_fmt_, cpu_engine);
    
    // do forward
    // for max pooling, need to return workspace
    if (pp->algo_kind == pooling_param_t::algorithm::pooling_max) {
        Tensor *ws_tensor = new Tensor((pooling2d_forward->ws_dims_), src->cxx_data_type(), pooling2d_forward->ws_fmt_, cpu_engine);

        pooling2d_forward->execute(src_tmp, dst_tensor->data(), ws_tensor->data());
        outputs.push_back(dst_tensor);
        outputs.push_back(ws_tensor);
    } else {
        pooling2d_forward->execute(src_tmp, dst_tensor->data());
        outputs.push_back(dst_tensor);
    }

    //FIXME here may cause performance issue
    if (src_reorder != NULL)
        delete static_cast<avx::byte *>(src_reorder);
    LOG(INFO) << "Succ exec pooling forward";
    return outputs;
}

template<typename T>
Tensor *Pooling2D<T>::Backward(
                Tensor *diff_dst,
                Tensor *ws,
                pooling_param_t *pp)
{
    return NULL;
    /*
    //sanity check
    mkldnn::memory::dims diff_src_dims = {cp->src_d1, cp->src_d2, cp->src_d3, cp->src_d4};
    mkldnn::memory::dims w_dims = {cp->weights_d1, cp->weights_d2, cp->weights_d3, cp->weights_d4};
    mkldnn::memory::dims diff_dst_dims = {cp->dst_d1, cp->dst_d2, cp->dst_d3, cp->dst_d4};
    assert(w_dims == weights->cxx_dims() && diff_dst_dims == diff_dst->cxx_dims());

    // sanity check for data type
    // assuem all x/w/b should have same data type as T
    // FIXME
    // yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == weights->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    // get a conv2d bwd data from primitive pool
    Convolution2DBwdData<T> *conv2d_bwd_data = NULL;
    conv2d_bwd_data = Convolution2DBwdDataFactory<T>::get( diff_src_dims, w_dims, diff_dst_dims,
            cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);

    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    mkldnn::memory::format w_fmt = weights->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();
    
    void* w_tmp = weights->data();
    void* diff_dst_tmp = diff_dst->data();
    void* w_reorder = NULL;
    void* diff_dst_reorder = NULL;

    if (w_fmt == conv2d_bwd_data->weights_fmt_ && diff_dst_fmt == conv2d_bwd_data->diff_dst_fmt_) {
        LOG(INFO) << "conv2d bwd data primitive fmt matched";
    } else {
        LOG(INFO) << "conv2d bwd data fmt not match, need to reorder";

        if (w_fmt != conv2d_bwd_data->weights_fmt_) {
            LOG(INFO) << "weight_fmt=" << w_fmt << ", conv2d_bwd_data->weights_fmt_="<< conv2d_bwd_data->weights_fmt_;
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, conv2d_bwd_data->weights_fmt_);
            w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder);
            w_tmp = w_reorder;
        } 
        if (diff_dst_fmt != conv2d_bwd_data->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_data->diff_dst_fmt_=" << conv2d_bwd_data->diff_dst_fmt_;
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, conv2d_bwd_data->diff_dst_fmt_);
            diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder);
            diff_dst_tmp = diff_dst_reorder;
        }
    }

    // create tensor based on selected primitive
    // assume dst and src have same data type
    Tensor *diff_src_tensor = new Tensor(diff_src_dims, diff_dst->cxx_data_type(), conv2d_bwd_data->diff_src_fmt_, cpu_engine);
    
    conv2d_bwd_data->execute(diff_src_tensor->data(), w_tmp, diff_dst_tmp);

    // free
    if (w_reorder != NULL)
        delete static_cast<avx::byte *>(w_reorder);
    if (diff_dst_reorder != NULL)
        delete static_cast<avx::byte *>(diff_dst_reorder);

    return diff_src_tensor;
    */
}


template class Pooling2D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
