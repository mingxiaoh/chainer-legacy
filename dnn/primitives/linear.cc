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
#include "linear.h"
#include "utils.h"
#include "linear_fwd.h"
#include "linear_bwd_data.h"
#include "linear_bwd_weights.h"
#include "linear_fwd_factory.h"
#include "linear_bwd_data_factory.h"
#include "linear_bwd_weights_factory.h"
#include "reorder_op.h"
#include "reorder_factory.h"
using namespace mkldnn;

extern const mkldnn::memory::dims NONE_DIMS; 
extern engine cpu_engine;

template<typename T>
Linear<T>::Linear()
{
}

template<typename T>
Linear<T>::~Linear()
{
}



template<typename T>
Tensor *Linear<T>::Forward(
        Tensor *src, Tensor *weights,
        Tensor *bias,
        linear_param_t *lp)
{
    //sanity check
    mkldnn::memory::dims src_dims = src->cxx_dims();
    mkldnn::memory::dims w_dims = weights->cxx_dims();
    mkldnn::memory::dims b_dims;
    mkldnn::memory::dims dst_dims ;
    if (lp->with_bias) {
        b_dims = bias->cxx_dims();
        assert(b_dims == bias->cxx_dims());
    }

    if (src->ndims() != weights->ndims()) {
        assert(weights->ndims() == 2 && src->ndims() == 4);
        w_dims = {w_dims[0], src_dims[1], src_dims[2], src_dims[3]};
        weights->reset_memory(format_2_as_4(weights->format()), w_dims);
    }
    dst_dims = {src_dims[0], w_dims[0]};

    //sanity check for data type
    //FIXME
    //is it possible y and w have different data type?
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == weights->cxx_data_type());
    if (lp->with_bias) {
        assert(memory_data_type<T>() == bias->cxx_data_type());
    }
    //get a linear from primitive pool
    LinearFwd<T> *linear_forward = NULL;
    if (lp->with_bias)
        linear_forward = LinearFwdFactory<T>::get(src_dims, w_dims, b_dims, dst_dims);
    else 
        linear_forward = LinearFwdFactory<T>::get(src_dims, w_dims, NONE_DIMS, dst_dims);
    //FIXME: in this model, every call to conv_forward will create a new mdarray, when to free?
    mkldnn::memory::format src_fmt = src->cxx_format();
    mkldnn::memory::format w_fmt = weights->cxx_format();
    void *src_tmp = src->data();
    void *w_tmp = weights->data();
    void *src_reorder = NULL;
    void *w_reorder = NULL;
    //check wheter format is match
    if(src_fmt == linear_forward->src_fmt_ && w_fmt == linear_forward->weights_fmt_) {
        //LOG(INFO) << "primitive fmt matched";
    } else {
        //LOG(INFO) << "format not matched, need to do reorder";
        if (src_fmt != linear_forward->src_fmt_) {
            LOG(INFO) << "src_fmt" << src_fmt << ", linear_forward->src_fmt_" << linear_forward->src_fmt_;
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, linear_forward->src_fmt_);
            src_reorder =  new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder);
            src_tmp = src_reorder;
        }
        if (w_fmt != linear_forward->weights_fmt_) {
            LOG(INFO) << "weight_fmt  = "<< w_fmt << ", linear_forward->weights_fmt_=" << linear_forward->weights_fmt_;
            //FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, linear_forward->weights_fmt_);
            w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder);
            w_tmp = w_reorder;
        }
    }
    //create mdarray based on primitive's dst
    Tensor *dst_tensor = new Tensor(dst_dims, src->cxx_data_type(), linear_forward->dst_fmt_, cpu_engine);
    // do forward
    if (lp->with_bias) {
        linear_forward->execute(src_tmp, w_tmp, bias->data(), dst_tensor->data());
    } else {
        linear_forward->execute(src_tmp, w_tmp, dst_tensor->data());
    }

    //FIXME: here  may cause performance issue
    if (src_reorder != NULL) 
        delete static_cast<avx::byte *>(src_reorder);
    if (w_reorder != NULL)
        delete static_cast<avx::byte *>(w_reorder);
    return dst_tensor;
}

/*
 * gW = gy * x
 */
template<typename T>
std::vector<Tensor *> Linear<T>::BackwardWeights(
            Tensor *src, Tensor* diff_dst, linear_param_t *lp)
{
    std::vector<Tensor *> bwd_weight_vec;
    mkldnn::memory::dims src_dims = src->cxx_dims();
    mkldnn::memory::dims diff_dst_dims = diff_dst->cxx_dims();
    mkldnn::memory::dims diff_w_dims;
    mkldnn::memory::dims diff_b_dims;
    if (lp->with_bias) 
        diff_b_dims = {lp->bias_d1};
    if (src->ndims() == 4) {
        diff_w_dims = {diff_dst_dims[1], src_dims[1], src_dims[2], src_dims[3]};
    } else if (src->ndims() == 2){
        diff_w_dims = {diff_dst_dims[1], src_dims[1]};
    } else {
        LOG(INFO) << "Error:: src only support 2 dims or 4 dims";
    }
    // sanity check for data type
    // FIXME
    // is it possible y and w ave different data type?
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());
    //get a linear bwd weights from primitive pool
    LinearBwdWeights<T> *linear_bwd_weights = NULL;
    if (lp->with_bias) {
        linear_bwd_weights = LinearBwdWeightsFactory<T>::get(src_dims, diff_w_dims, diff_b_dims, diff_dst_dims);
    } else {
        linear_bwd_weights = LinearBwdWeightsFactory<T>::get(src_dims, diff_w_dims, NONE_DIMS, diff_dst_dims);
    }
    //create tensor based on selected primitive
    mkldnn::memory::format src_fmt = src->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();
    //assum dst and src have same data type
    void* src_tmp = src->data();
    void* diff_dst_tmp = diff_dst->data();
    void* src_reorder = NULL;
    void* diff_dst_reorder = NULL;
    //check whether fmt is same
    if (src_fmt == linear_bwd_weights->src_fmt_ && diff_dst_fmt == linear_bwd_weights->diff_dst_fmt_) {
        LOG(INFO) << "primitive fmt matched";
    } else {
        LOG(INFO) << "fmt not match, need to reorder";
        if (src_fmt != linear_bwd_weights->src_fmt_) {
            LOG(INFO) <<  "src_fmt = " << src_fmt << ", linear_bwd_weights->src_fmt_=" << linear_bwd_weights->src_fmt_;
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, linear_bwd_weights->src_fmt_);
            src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder);
            src_tmp = src_reorder;
        }
        if (diff_dst_fmt != linear_bwd_weights->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt = " << diff_dst_fmt << ", linear_bwd_weights->diff_dst_fmt = " << linear_bwd_weights->diff_dst_fmt_;
            //FIXME when to free the reordered memory
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, linear_bwd_weights->diff_dst_fmt_);
            diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder);
            diff_dst_tmp = diff_dst_reorder;
        }
    }
    //assume dst and src have the same data type
    Tensor *diff_w_tensor = new Tensor(diff_w_dims, src->cxx_data_type(), linear_bwd_weights->diff_weights_fmt_, cpu_engine);
    //do execute
    if(lp->with_bias) {
        //assume bias's format is always mkldnn::memory::format::x
        Tensor *diff_b_tensor = new Tensor(diff_b_dims, src->cxx_data_type(), mkldnn::memory::format::x, cpu_engine);
        linear_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_b_tensor->data(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_tensor);
        bwd_weight_vec.push_back(diff_b_tensor);
    } else {
        linear_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_tensor);
    }
    //free
    if(src_reorder != NULL)
        delete static_cast<avx::byte *>(src_reorder);
    if(diff_dst_reorder != NULL)
        delete static_cast<avx::byte *>(diff_dst_reorder);
    return bwd_weight_vec;
}

template<typename T>
Tensor *Linear<T>::BackwardData(
            Tensor *weights, Tensor *diff_dst,
            linear_param_t *lp)
{
    //sanity check
    mkldnn::memory::dims w_dims = weights->cxx_dims();
    mkldnn::memory::dims diff_dst_dims = diff_dst->cxx_dims();
    mkldnn::memory::dims diff_src_dims;
    if (lp->src_ndims == 2) {
        assert(weights->ndims() == 2);
        diff_src_dims = {lp->src_d1, lp->src_d2};
    } else if (lp->src_ndims == 4) {
        diff_src_dims = {lp->src_d1, lp->src_d2, lp->src_d3, lp->src_d4};
        if (weights->ndims() != 4) {
            w_dims = {w_dims[0], diff_src_dims[1], diff_src_dims[2], diff_src_dims[3]};
            weights->reset_memory(format_2_as_4(weights->format()), w_dims);
        }
    } else {
        LOG(INFO) << "Error:: src ndim not support(2 or 4 only)";
    }
    //sanity check for data type
    //assume all a/w/b should have the same type as T
    //FIXME
    //is it possible x and w have different data type???
    assert(memory_data_type<T>() == weights->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());
    //get a linear bwd  data from primitive pool
    LinearBwdData<T> *linear_bwd_data = NULL;
    linear_bwd_data = LinearBwdDataFactory<T>::get(diff_src_dims, w_dims, diff_dst_dims);
    //FIXME: in this model, every call to linear_forward will create a new tensor, when to free??
    mkldnn::memory::format w_fmt = weights->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();
    
    void* w_tmp = weights->data();
    void* diff_dst_tmp = diff_dst->data();
    void* w_reorder = NULL;
    void* diff_dst_reorder = NULL;

    if (w_fmt == linear_bwd_data->weights_fmt_ && diff_dst_fmt == linear_bwd_data->diff_dst_fmt_) {
        LOG(INFO) << "linear bwd data primitive fmt matched";
    } else {
        LOG(INFO) << "linear bwd data fmt not match, need to reorder";
        if (w_fmt != linear_bwd_data->weights_fmt_) {
            LOG(INFO) << "weights_fmt_ = " << w_fmt << ", linear_bwd_data->weights_fmt_ = " << linear_bwd_data->weights_fmt_;
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, linear_bwd_data->weights_fmt_);
            w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder);
            w_tmp = w_reorder;
        }
        if (diff_dst_fmt != linear_bwd_data->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt = " << diff_dst_fmt << ", linear_bwd_data->diff_dst_fmt = " << linear_bwd_data->diff_dst_fmt_;
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, linear_bwd_data->diff_dst_fmt_);
            diff_dst_reorder  = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder);
            diff_dst_tmp = diff_dst_reorder;
        }
    }
    //create tensor based on selected primitive
    //assume dst and src have the same data type
    Tensor* diff_src_tensor = new Tensor(diff_src_dims, diff_dst->cxx_data_type(), linear_bwd_data->diff_src_fmt_, cpu_engine);
    linear_bwd_data->execute(diff_src_tensor->data(), w_tmp, diff_dst_tmp);
    // free
    if (w_reorder != NULL)
        delete static_cast<avx::byte *>(w_reorder);
    if (diff_dst_reorder != NULL)
        delete static_cast<avx::byte *>(diff_dst_reorder);
    return diff_src_tensor;
}
template class Linear<float>;

