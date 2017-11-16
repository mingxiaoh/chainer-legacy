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
#include "conv.h"
#include "utils.h"
#include "conv_fwd.h"
#include "conv_bwd_data.h"
#include "conv_bwd_weights.h"
#include "prim_factory.h"
#include "reorder_op.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2D<T>::Convolution2D()
{
}

template<typename T>
Convolution2D<T>::~Convolution2D()
{
}

template<typename T>
Tensor *Convolution2D<T>::Forward(
                Tensor *src, Tensor *weights,
                Tensor *bias,
                conv_param_t *cp)
{
    // sanity check
    mkldnn::memory::dims src_dims = {cp->src_d1, cp->src_d2, cp->src_d3, cp->src_d4};
    mkldnn::memory::dims w_dims = {cp->weights_d1, cp->weights_d2, cp->weights_d3, cp->weights_d4};
    mkldnn::memory::dims dst_dims = {cp->dst_d1, cp->dst_d2, cp->dst_d3, cp->dst_d4};
    mkldnn::memory::dims b_dims;
    if (cp->with_bias) {
        b_dims = {cp->bias_d1};
        assert( b_dims == bias->cxx_dims());
    }
    assert(src_dims == src->cxx_dims() && w_dims = weights-.cxx_dims());

    //sanity check for data type
    //assuem all x/w/b should have same data type as T
    //FIXME
    //yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == weights->cxx_data_type());
    if (cp->with_bias)
        assert(memory_data_type<T>() == bias->cxx_data_type());
    
    // get a conv2d fwd from primitive pool
    Convolution2DFwd<T> *conv2d_forward = NULL;
    if (cp->with_bias)
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, b_dims, dst_dims,
                cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    else
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, NONE_DIMS, dst_dims,
                cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    
    // FIXME: in this model, every call to conv_forward will create a new tensor, when to free???
    mkldnn::memory::format src_fmt = src->cxx_format(); // src fmt in tensor
    mkldnn::memory::format w_fmt = weights->cxx_format(); // weight fmt in tensor

    void *src_tmp = src->data();
    void *w_tmp = weights->data();
    void *src_reorder = NULL;
    void *w_reorder = NULL;
    
    // check wehther fmt is same
    if (src_fmt == conv2d_forward->src_fmt_ && w_fmt == conv2d_forward->weights_fmt_) {
        LOG(INFO) << "primitive fmt matched";
    } else {
        LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_forward->src_fmt_) {
            LOG(INFO) << "src_fmt=" << src_fmt <<", conv2d_forward->src_fmt_=" << conv2d_forward->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, conv2d_forward->src_fmt_);
            src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder);
            src_tmp = src_reorder;
        }

        if (w_fmt != conv2d_forward->weights_fmt_) {
            LOG(INFO) << "weight_fmt=" << w_fmt <<", conv2d_forward->weight_fmt_=" << conv2d_forward->weights_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_w_op = ReorderFactory<T>::get(w_dims, w_fmt, conv2d_forward->weights_fmt_);
            w_reorder = new avx::byte[weights->len()];
            reorder_w_op->execute(w_tmp, w_reorder);
            w_tmp = w_reorder;
            
            
            // set internal fmt back to weight tensor
            if ( cp->with_weights_opt ) {
                weights->reset_memory(
                        static_cast<mkldnn_memory_format_t>(conv2d_forward->weights_fmt_),
                        static_cast<avx::byte *>(w_reorder));
            }
        }
    }

    // create tensor based on primitive's dst 
    // assume dst and src have same data type
    Tensor *dst_tensor = new Tensor(dst_dims, src->cxx_data_type(), conv2d_forward->dst_fmt_, cpu_engine);
    
    // do forward
    if (cp->with_bias) {
        conv2d_forward->execute(src_tmp, w_tmp, bias->data(), dst_tensor->data());
    } else {
        conv2d_forward->execute(src_tmp, w_tmp, dst_tensor->data());
    }

    //FIXME here may cause performance issue
    if (src_reorder != NULL)
        delete static_cast<avx::byte *>(src_reorder);
    if ( !cp->with_weights_opt && w_reorder != NULL)
        delete static_cast<avx::byte *>(w_reorder);

    return dst_tensor;
}

/*
 * gW = gy *x
 */
template<typename T>
std::vector<Tensor *> Convolution2D<T>::BackwardWeights(
                Tensor *src, Tensor *diff_dst,
                conv_param_t *cp)
{
    std::vector<Tensor *> bwd_weight_vec;

    // sanity check
    mkldnn::memory::dims src_dims = {cp->src_d1, cp->src_d2, cp->src_d3, cp->src_d4};
    mkldnn::memory::dims diff_w_dims = {cp->weights_d1, cp->weights_d2, cp->weights_d3, cp->weights_d4};
    mkldnn::memory::dims diff_dst_dims = {cp->dst_d1, cp->dst_d2, cp->dst_d3, cp->dst_d4};
    mkldnn::memory::dims diff_b_dims;
    if (cp->with_bias)
        diff_b_dims = {cp->bias_d1};
    assert(src_dims == src->cxx_dims() && diff_dst_dims = diff_dst->cxx_dims());

    // sanity check for data type
    // FIXME
    // is it possible y and w have different data type??
    assert(memory_data_type<T>() == src->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst->cxx_data_type());

    // get a conv2d bwd weights from primitive pool
    Convolution2DBwdWeights<T> *conv2d_bwd_weights = NULL;
    if (cp->with_bias) {
        conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, diff_b_dims, diff_dst_dims, 
                cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    } else {
        conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, NONE_DIMS, diff_dst_dims, 
                cp->sy, cp->sx, cp->pad_lh, cp->pad_lw, cp->pad_rh, cp->pad_rw);
    }

    // create tensor based on selected primitive
    mkldnn::memory::format src_fmt = src->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst->cxx_format();

    //assum dst and src have same data type
    void* src_tmp = src->data();
    void* diff_dst_tmp = diff_dst->data();
    void* src_reorder = NULL;
    void* diff_dst_reorder = NULL;

    //check whether fmt is same
    if (src_fmt == conv2d_bwd_weights->src_fmt_ && diff_dst_fmt == conv2d_bwd_weights->diff_dst_fmt_) {
        LOG(INFO) << "primitive fmt matched";
    } else {
        LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_bwd_weights->src_fmt_) {
            LOG(INFO) << "src_fmt=" << src_fmt << ", conv2d_bwd_weights->src_fmt_=" << conv2d_bwd_weights->src_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_src_op = ReorderFactory<T>::get(src_dims, src_fmt, conv2d_bwd_weights->src_fmt_);
            src_reorder = new avx::byte[src->len()];
            reorder_src_op->execute(src_tmp, src_reorder);
            src_tmp = src_reorder;
        }
        if (diff_dst_fmt != conv2d_bwd_weights->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_weights->diff_dst_fmt_=" << conv2d_bwd_weights->diff_dst_fmt_;
            // FIXME: when to free the reordered memory
            ReorderOp<T>* reorder_diff_dst_op = ReorderFactory<T>::get(diff_dst_dims, diff_dst_fmt, conv2d_bwd_weights->diff_dst_fmt_);
            diff_dst_reorder = new avx::byte[diff_dst->len()];
            reorder_diff_dst_op->execute(diff_dst_tmp, diff_dst_reorder);
            diff_dst_tmp = diff_dst_reorder;
        }
    }

    //assum dst and src have same data type
    Tensor *diff_w_tensor = new Tensor(diff_w_dims, src->cxx_data_type(), conv2d_bwd_weights->diff_weights_fmt_, cpu_engine);
        // do execute
    if (cp->with_bias) {
        // asume bias's format is always mkldnn::memory::format::x
        Tensor *diff_b_tensor = new Tensor(diff_b_dims, src->cxx_data_type(), mkldnn::memory::format::x, cpu_engine);
        conv2d_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_b_tensor->data(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_tensor);
        bwd_weight_vec.push_back(diff_b_tensor);
    } else {
        conv2d_bwd_weights->execute(src_tmp, diff_w_tensor->data(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_tensor);
    }

    //free
    if (src_reorder != NULL)
        delete static_cast<avx::byte *>(src_reorder);
    if (diff_dst_reorder != NULL)
        delete static_cast<avx::byte *>(diff_dst_reorder);
    return bwd_weight_vec;
}

template<typename T>
Tensor *Convolution2D<T>::BackwardData(
                Tensor *weights, Tensor *diff_dst,
                conv_param_t *cp)
{
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
}


template class Convolution2D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
