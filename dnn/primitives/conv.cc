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
#include "conv.h"
#include "utils.h"
#include "conv_fwd.h"
#include "conv_bwd_data.h"
#include "conv_bwd_weights.h"
#include "conv_fwd_factory.h"
#include "conv_bwd_data_factory.h"
#include "conv_bwd_weights_factory.h"

using namespace mkldnn;

const mkldnn::memory::dims NONE_DIMS = {}; 
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
mdarray Convolution2D<T>::Forward(
                mdarray &src, mdarray &weights,
                mdarray &bias,
                conv_param_t &cp)
{
    //get internal mdarray buffer
    implementation::mdarray *src_internal = src.get();
    implementation::mdarray *w_internal = weights.get();
    implementation::mdarray *b_internal;
    if (cp.with_bias)
        b_internal = bias.get();
    
    // sanity check
    mkldnn::memory::dims src_dims = {cp.src_d1, cp.src_d2, cp.src_d3, cp.src_d4};
    mkldnn::memory::dims w_dims = {cp.weights_d1, cp.weights_d2, cp.weights_d3, cp.weights_d4};
    mkldnn::memory::dims dst_dims = {cp.dst_d1, cp.dst_d2, cp.dst_d3, cp.dst_d4};
    mkldnn::memory::dims b_dims;
    if (cp.with_bias) {
        b_dims = {cp.bias_d1};
        assert( b_dims == b_internal->cxx_dims());
    }
    assert(src_dims == src_internal->cxx_dims() && w_dims = w_internal->cxx_dims());

    //sanity check for data type
    //assuem all x/w/b should have same data type as T
    //FIXME
    //yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == src_internal->cxx_data_type());
    assert(memory_data_type<T>() == w_internal->cxx_data_type());
    if (cp.with_bias)
        assert(memory_data_type<T>() == b_internal->cxx_data_type());
    
    // get a conv2d fwd from primitive pool
    Convolution2DFwd<T> *conv2d_forward = NULL;
    if (cp.with_bias)
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, b_dims, dst_dims,
                cp.sy, cp.sx, cp.pad_lh, cp.pad_lw, cp.pad_rh, cp.pad_rw);
    else
        conv2d_forward = Convolution2DFwdFactory<T>::get(src_dims, w_dims, NONE_DIMS, dst_dims,
                cp.sy, cp.sx, cp.pad_lh, cp.pad_lw, cp.pad_rh, cp.pad_rw);
    
    // FIXME: in this model, every call to conv_forward will create a new mdarray, when to free???
    mkldnn::memory::format src_fmt = src_internal->cxx_format(); // src fmt in mdarray
    mkldnn::memory::format w_fmt = w_internal->cxx_format(); // weight fmt in mdarray

    mkldnn::memory src_tmp = src_internal->memory(); // memory in src mdarray
    mkldnn::memory w_tmp = w_internal->memory(); // memory in weight mdarray
    std::shared_ptr<mkldnn::memory> src_reorder = NULL;
    std::shared_ptr<mkldnn::memory> w_reorder = NULL;
    
    // check wehther fmt is same
    if (src_fmt == conv2d_forward->src_fmt_ && w_fmt == conv2d_forward->weights_fmt_) {
        LOG(INFO) << "primitive fmt matched";
    } else {
        LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_forward->src_fmt_) {
            LOG(INFO) << "src_fmt=" << src_fmt <<", conv2d_forward->src_fmt_=" << conv2d_forward->src_fmt_;
            // FIXME: when to free the reordered memory
            src_reorder.reset(new memory({{{src_dims}, memory_data_type<T>(), conv2d_forward->src_fmt_}, cpu_engine}));
            reorder_func(src_internal->memory(), *(src_reorder));
            src_tmp = *src_reorder;
        }

        if (w_fmt != conv2d_forward->weights_fmt_) {
            LOG(INFO) << "weight_fmt=" << w_fmt <<", conv2d_forward->weight_fmt_=" << conv2d_forward->weights_fmt_;
            // FIXME: when to free the reordered memory
            w_reorder.reset(new memory({{{w_dims}, memory_data_type<T>(), conv2d_forward->weights_fmt_}, cpu_engine}));
            reorder_func(w_internal->memory(), *(w_reorder));
            w_tmp = *w_reorder;
        }
    }

    // create mdarray based on primitive's dst 
    // assume dst and src have same data type
    mdarray dst_mdarray = mdarray(dst_dims, src_internal->cxx_data_type(), conv2d_forward->dst_fmt_, cpu_engine);
    
    // do forward
    if (cp.with_bias) {
        conv2d_forward->execute(src_tmp, w_tmp, b_internal->memory(), dst_mdarray.get()->memory());
    } else {
        conv2d_forward->execute(src_tmp, w_tmp, dst_mdarray.get()->memory());
    }
    //

    return dst_mdarray;
}

/*
 * gW = gy *x
 */
template<typename T>
std::vector<mdarray> Convolution2D<T>::BackwardWeights(
                mdarray &src, mdarray &diff_dst,
                conv_param_t &cp)
{
    std::vector<mdarray> bwd_weight_vec;

    // get internal mdarray buffer
    implementation::mdarray *src_internal = src.get();
    implementation::mdarray *diff_dst_internal = diff_dst.get();

    // sanity check
    mkldnn::memory::dims src_dims = {cp.src_d1, cp.src_d2, cp.src_d3, cp.src_d4};
    mkldnn::memory::dims diff_w_dims = {cp.weights_d1, cp.weights_d2, cp.weights_d3, cp.weights_d4};
    mkldnn::memory::dims diff_dst_dims = {cp.dst_d1, cp.dst_d2, cp.dst_d3, cp.dst_d4};
    mkldnn::memory::dims diff_b_dims;
    if (cp.with_bias)
        diff_b_dims = {cp.bias_d1};
    assert(src_dims == src_internal->cxx_dims() && diff_dst_dims = diff_dst_internal->cxx_dims());

    // sanity check for data type
    // FIXME
    // is it possible y and w have different data type??
    assert(memory_data_type<T>() == src_internal->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst_internal->cxx_data_type());

    // get a conv2d bwd weights from primitive pool
    Convolution2DBwdWeights<T> *conv2d_bwd_weights = NULL;
    if (cp.with_bias) {
        conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, diff_b_dims, diff_dst_dims, 
                cp.sy, cp.sx, cp.pad_lh, cp.pad_lw, cp.pad_rh, cp.pad_rw);
    } else {
        conv2d_bwd_weights = Convolution2DBwdWeightsFactory<T>::get(src_dims, diff_w_dims, NONE_DIMS, diff_dst_dims, 
                cp.sy, cp.sx, cp.pad_lh, cp.pad_lw, cp.pad_rh, cp.pad_rw);
    }

    // create mdarray based on selected primitive
    mkldnn::memory::format src_fmt = src_internal->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst_internal->cxx_format();

    //assum dst and src have same data type
    mkldnn::memory src_tmp = src_internal->memory();
    mkldnn::memory diff_dst_tmp = diff_dst_internal->memory();
    std::shared_ptr<mkldnn::memory> src_reorder = NULL;
    std::shared_ptr<mkldnn::memory> diff_dst_reorder = NULL;

    //check whether fmt is same
    if (src_fmt == conv2d_bwd_weights->src_fmt_ && diff_dst_fmt == conv2d_bwd_weights->diff_dst_fmt_) {
        LOG(INFO) << "primitive fmt matched";
    } else {
        LOG(INFO) << "fmt not match, need to reorder";

        if (src_fmt != conv2d_bwd_weights->src_fmt_) {
            LOG(INFO) << "src_fmt=" << src_fmt << ", conv2d_bwd_weights->src_fmt_=" << conv2d_bwd_weights->src_fmt_;
            // FIXME: when to free the reordered memory
            src_reorder.reset(new memory({{{src_dims}, memory_data_type<T>(), conv2d_bwd_weights->src_fmt_}, cpu_engine}));
            reorder_func(src_internal->memory(), *(src_reorder));
            src_tmp = *src_reorder;
        }
        if (diff_dst_fmt != conv2d_bwd_weights->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_weights->diff_dst_fmt_=" << conv2d_bwd_weights->diff_dst_fmt_;
            // FIXME: when to free the reordered memory
            diff_dst_reorder.reset(new memory({{{diff_dst_dims}, memory_data_type<T>(), conv2d_bwd_weights->diff_dst_fmt_}, cpu_engine}));
            reorder_func(diff_dst_internal->memory(), *(diff_dst_reorder));
            diff_dst_tmp = *diff_dst_reorder;
        }
    }

    //assum dst and src have same data type
    mdarray diff_w_mdarray = mdarray(diff_w_dims, src_internal->cxx_data_type(), conv2d_bwd_weights->diff_weights_fmt_, cpu_engine);
        // do execute
    if (cp.with_bias) {
        // asume bias's format is always mkldnn::memory::format::x
        mdarray diff_b_mdarray = mdarray(diff_b_dims, src_internal->cxx_data_type(), mkldnn::memory::format::x, cpu_engine);
        conv2d_bwd_weights->execute(src_tmp, diff_w_mdarray.get()->memory(), diff_b_mdarray.get()->memory(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_mdarray);
        bwd_weight_vec.push_back(diff_b_mdarray);
    } else {
        conv2d_bwd_weights->execute(src_tmp, diff_w_mdarray.get()->memory(), diff_dst_tmp);
        bwd_weight_vec.push_back(diff_w_mdarray);
    }
    return bwd_weight_vec;
}

template<typename T>
mdarray Convolution2D<T>::BackwardData(
                mdarray &weights, mdarray &diff_dst,
                conv_param_t &cp)
{
    // get internal mdarray buffer
    implementation::mdarray *w_internal = weights.get();
    implementation::mdarray *diff_dst_internal = diff_dst.get();

    //sanity check
    mkldnn::memory::dims diff_src_dims = {cp.src_d1, cp.src_d2, cp.src_d3, cp.src_d4};
    mkldnn::memory::dims w_dims = {cp.weights_d1, cp.weights_d2, cp.weights_d3, cp.weights_d4};
    mkldnn::memory::dims diff_dst_dims = {cp.dst_d1, cp.dst_d2, cp.dst_d3, cp.dst_d4};
    assert(w_dims == w_internal->cxx_dims() && diff_dst_dims == diff_dst_internal->cxx_dims());

    // sanity check for data type
    // assuem all x/w/b should have same data type as T
    // FIXME
    // yli135: Is it possible x and w have different data type????
    assert(memory_data_type<T>() == w_internal->cxx_data_type());
    assert(memory_data_type<T>() == diff_dst_internal->cxx_data_type());

    // get a conv2d bwd data from primitive pool
    Convolution2DBwdData<T> *conv2d_bwd_data = NULL;
    conv2d_bwd_data = Convolution2DBwdDataFactory<T>::get( diff_src_dims, w_dims, diff_dst_dims,
            cp.sy, cp.sx, cp.pad_lh, cp.pad_lw, cp.pad_rh, cp.pad_rw);

    // FIXME: in this model, every call to conv_forward will create a new mdarray, when to free???
    mkldnn::memory::format w_fmt = w_internal->cxx_format();
    mkldnn::memory::format diff_dst_fmt = diff_dst_internal->cxx_format();
    
    mkldnn::memory w_tmp = w_internal->memory();
    mkldnn::memory diff_dst_tmp = diff_dst_internal->memory();
    std::shared_ptr<mkldnn::memory> w_reorder = NULL;
    std::shared_ptr<mkldnn::memory> diff_dst_reorder = NULL;

    if (w_fmt == conv2d_bwd_data->weights_fmt_ && diff_dst_fmt == conv2d_bwd_data->diff_dst_fmt_) {
        LOG(INFO) << "conv2d bwd data primitive fmt matched";
    } else {
        LOG(INFO) << "conv2d bwd data fmt not match, need to reorder";

        if (w_fmt != conv2d_bwd_data->weights_fmt_) {
            LOG(INFO) << "weight_fmt=" << w_fmt << ", conv2d_bwd_data->weights_fmt_="<< conv2d_bwd_data->weights_fmt_;
            w_reorder.reset(new memory({{{w_dims}, memory_data_type<T>(), conv2d_bwd_data->weights_fmt_}, cpu_engine}));
            reorder_func(w_internal->memory(), *(w_reorder));
            w_tmp = *w_reorder;
        } 
        if (diff_dst_fmt != conv2d_bwd_data->diff_dst_fmt_) {
            LOG(INFO) << "diff_dst_fmt=" << diff_dst_fmt <<", conv2d_bwd_data->diff_dst_fmt_=" << conv2d_bwd_data->diff_dst_fmt_;
            diff_dst_reorder.reset(new memory({{{diff_dst_dims}, memory_data_type<T>(), conv2d_bwd_data->diff_dst_fmt_}, cpu_engine}));
            reorder_func(diff_dst_internal->memory(), *(diff_dst_reorder));
            diff_dst_tmp = *diff_dst_reorder;
        }
    }

    // create mdarray based on selected primitive
    // assume dst and src have same data type
    mdarray diff_src_mdarray = mdarray(diff_src_dims, diff_dst_internal->cxx_data_type(), conv2d_bwd_data->diff_src_fmt_, cpu_engine);
    
    conv2d_bwd_data->execute(diff_src_mdarray.get()->memory(), w_tmp, diff_dst_tmp);

    return diff_src_mdarray;
}


template class Convolution2D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
