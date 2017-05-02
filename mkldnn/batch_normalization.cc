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
#include "batch_normalization.h"
#include "utils.h"

using namespace mkldnn;


template<typename T>
BatchNormalization<T>::BatchNormalization(double eps,
                                          bool is_training,
                                          bool has_weights,
                                          bool fixed_mean_var) {
    eps_ = eps;
    if (has_weights)
        flags_ |= use_scale_shift;
    if (fixed_mean_var)
        flags_ |= use_global_stats;
    fwd_prop_kind_ = is_training ? prop_kind::forward_training : prop_kind::forward_scoring;

    eng_.reset(new engine(engine::kind::cpu, 0));
}

template<typename T>
void BatchNormalization<T>::forward_setup(
    int x_d1, int x_d2, int x_d3, int x_d4,
    int y_d1, int y_d2, int y_d3, int y_d4,
    int W_d1, int W_d2, int mean_d1, int var_d1)
{
    /* check AVX512 first then AVX2 */
    memory::format fmt_desired;
    if (cpu_support_avx512_p() && (x_d2%16) == 0) {
        fmt_desired = memory::format::nChw16c;
        LOG(INFO) << "forward_setup nChw16c";
    } else if (cpu_support_avx2_p() && (x_d2%8) == 0) {
        fmt_desired = memory::format::nChw8c;
        LOG(INFO) << "forward_setup nChw8c";
    } else {
        fmt_desired = memory::format::nchw;
    }

    memory::dims src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims dst_tz = {y_d1, y_d2, y_d3, y_d4};

    assert(W_d1 == 2);
    assert(x_d2 == y_d2 == W_d2 == mean_d1 == var_d1);

    /* create memory for user data */
    LOG(INFO) << "create memory for user data";
    user_src_mem_.reset(new memory({{{src_tz}, memory_data_type<T>(),
                                   memory::format::nchw}, *eng_}, dummy));
    auto src_md =  memory::desc({src_tz}, memory_data_type<T>(), fmt_desired);

    user_dst_mem_.reset(new memory({{{dst_tz}, memory_data_type<T>(),
                                   memory::format::nchw }, *eng_}, dummy));

    if (!fwd_pd_)
    {
        LOG(INFO) << "fwd_desc_";
        fwd_desc_.reset(new batch_normalization_forward::desc(
                fwd_prop_kind_, src_md, eps_, flags_));
        fwd_pd_.reset(new batch_normalization_forward::primitive_desc(
                *fwd_desc_, *eng_));
    }

    src_mem_ = user_src_mem_;
    dst_mem_ = user_dst_mem_;
    bool reorder_src_p = false;
    bool reorder_dst_p = false;

    if (fmt_desired != memory::format::nchw) {
        src_mem_.reset(new memory({{{src_tz}, memory_data_type<T>(), fmt_desired}, *eng_}));
        reorder_src_.reset(new reorder(*user_src_mem_, *src_mem_));
        reorder_src_p = true;
    }

    if (memory::primitive_desc(fwd_pd_.get()->dst_primitive_desc())
        != user_dst_mem_->get_primitive_desc()) {
        dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc()));
        reorder_dst_.reset(new reorder(*dst_mem_, *user_dst_mem_));
        reorder_dst_p = true;
    }

    bool scale_shift = flags_ & use_scale_shift;
    bool global_stats = flags_ & use_global_stats;
    bool is_training = fwd_prop_kind_ == prop_kind::forward_training;

    if (scale_shift)
        weights_mem_.reset(new memory(fwd_pd_->weights_primitive_desc(), dummy));

    if (is_training || global_stats) {
        mean_mem_.reset(new memory(fwd_pd_->mean_primitive_desc(), dummy));
        var_mem_.reset(new memory(fwd_pd_->variance_primitive_desc(), dummy));
    }

    if (!is_training && !global_stats) {
        if (scale_shift) {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, *weights_mem_, *dst_mem_));
        } else {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, *dst_mem_));
        }
    } else if (global_stats) {
        if (scale_shift) {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, (const primitive::at)*mean_mem_,
                    (const primitive::at)*var_mem_, *weights_mem_, *dst_mem_));
        } else {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, (const primitive::at)*mean_mem_,
                    (const primitive::at)*var_mem_, *dst_mem_));
        }
    } else {
        if (scale_shift) {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, *weights_mem_, *dst_mem_, *mean_mem_, *var_mem_));
        } else {
            batch_normalization_fwd_.reset(new batch_normalization_forward(
                    *fwd_pd_, *src_mem_, *dst_mem_, *mean_mem_, *var_mem_));
        }

    }

    if (reorder_src_p)
        this->fwd_primitives_.push_back(*reorder_src_);
    fwd_primitives_.push_back(*batch_normalization_fwd_);
    if (reorder_dst_p)
        this->fwd_primitives_.push_back(*reorder_dst_);
    fwd_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
void BatchNormalization<T>::fwd_reset_mem(T* x, T* y, T* W, T* mean, T* var)
{
    user_src_mem_->set_data_handle(x);
    user_dst_mem_->set_data_handle(y);
    mean_mem_->set_data_handle(mean);
    var_mem_->set_data_handle(var);
    if (flags_ & use_scale_shift)
        weights_mem_->set_data_handle(W);
}

template<typename T>
void BatchNormalization<T>::forward(
    T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
    T* y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
    T* W, int W_d1, int W_d2,
    T* mean, int mean_d1, T* var,  int var_d1)
{
    if (forward_first_use_) {
        LOG(INFO) << "forward forward_first_use_";
        forward_first_use_ = false;
        if (!fwd_stream_){
            forward_setup(x_d1, x_d2, x_d3, x_d4,
                          y_d1, y_d2, y_d3, y_d4,
                          W_d1, W_d2, mean_d1, var_d1);
        }
        fwd_reset_mem(x, y, W, mean, var);
        fwd_stream_->submit(fwd_primitives_).wait();
    } else {
        fwd_reset_mem(x, y, W, mean, var);
        fwd_stream_->rerun().wait();
    }
}

template<typename T>
int BatchNormalization<T>::backward_setup(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var, int var_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gW, int gW_d1, int gW_d2)
{
    memory::format fmt_desired;
    // we check AVX512 first then AVX2
    if (cpu_support_avx512_p() && (x_d2%16)==0) {
        LOG(INFO) << "backward_setup nChw16c";
        fmt_desired = memory::format::nChw16c;
    } else if (cpu_support_avx2_p() && (x_d2%8)==0) {
        fmt_desired = memory::format::nChw8c;
        LOG(INFO) << "backward_setup nChw8c";
    } else {
        fmt_desired = memory::format::nchw;
    }

    memory::dims src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims diff_src_tz = {gx_d1, gx_d2, gx_d3, gx_d4};
    memory::dims diff_dst_tz = {gy_d1, gy_d2, gy_d3, gy_d4};

    bwd_user_src_mem_.reset(new memory({{{src_tz}, memory_data_type<T>(),
                                       memory::format::nchw}, *eng_}, dummy));
    user_diff_src_mem_.reset(new memory({{{diff_src_tz}, memory_data_type<T>(),
                                        memory::format::nchw}, *eng_}, dummy));
    user_diff_dst_mem_.reset(new memory({{{diff_dst_tz}, memory_data_type<T>(),
                                        memory::format::nchw}, *eng_}, dummy));

    auto bwd_src_md = memory::desc({src_tz}, memory_data_type<T>(), fmt_desired);
    auto diff_dst_md = memory::desc({diff_dst_tz}, memory_data_type<T>(), fmt_desired);

    bwd_desc_.reset(new batch_normalization_backward::desc(
            prop_kind::backward, diff_dst_md, bwd_src_md, eps_, flags_));
    bwd_pd_.reset(new batch_normalization_backward::primitive_desc(
            *bwd_desc_, *eng_, *fwd_pd_));

    bwd_src_mem_ = bwd_user_src_mem_;
    diff_src_mem_  = user_diff_src_mem_;
    diff_dst_mem_  = user_diff_dst_mem_;

    bool reorder_diff_dst_p = false;
    bool reorder_bwd_src_p = false;
    bool reorder_diff_src_p = false;

    if (fmt_desired != memory::format::nchw) {
        diff_dst_mem_.reset(new memory({{{diff_dst_tz}, memory_data_type<T>(), fmt_desired}, *eng_}));
        reorder_diff_dst_.reset(new reorder(*user_diff_dst_mem_, *diff_dst_mem_));
        reorder_diff_dst_p = true;

        bwd_src_mem_.reset(new memory({{{src_tz}, memory_data_type<T>(), fmt_desired}, *eng_}));
        reorder_bwd_src_.reset(new reorder(*bwd_user_src_mem_, *bwd_src_mem_));
        reorder_bwd_src_p = true;

        diff_src_mem_.reset(new memory({{{src_tz}, memory_data_type<T>(), fmt_desired}, *eng_}));
        reorder_diff_src_.reset(new reorder(*diff_src_mem_, *user_diff_src_mem_));
        reorder_diff_src_p = true;
    }

    if (flags_ & use_scale_shift) {
        diff_weights_mem_.reset(new memory(bwd_pd_->diff_weights_primitive_desc(), dummy));
        batch_normalization_bwd_.reset(new batch_normalization_backward(
                *bwd_pd_, *bwd_src_mem_, *mean_mem_, *var_mem_,
                *diff_dst_mem_, *weights_mem_, *diff_src_mem_, *diff_weights_mem_));
    } else {
        batch_normalization_bwd_.reset(new batch_normalization_backward(
                *bwd_pd_, *bwd_src_mem_, *mean_mem_, *var_mem_,
                *diff_dst_mem_, *diff_src_mem_));

    }

    if (reorder_diff_dst_p)
        bwd_primitives_.push_back(*reorder_diff_dst_);
    if (reorder_bwd_src_p)
        bwd_primitives_.push_back(*reorder_bwd_src_);
    bwd_primitives_.push_back(*batch_normalization_bwd_);
    if (reorder_diff_src_p)
        bwd_primitives_.push_back(*reorder_diff_src_);
    bwd_stream_.reset(new stream(stream::kind::eager));

    return 0;
}

template<typename T>
void BatchNormalization<T>::bwd_reset_mem(
    T* x, T* W, T* mean, T* var, T* gy, T* gx, T* gW)
{
    bwd_user_src_mem_->set_data_handle(x);
    mean_mem_->set_data_handle(mean);
    var_mem_->set_data_handle(var);
    user_diff_dst_mem_->set_data_handle(gy);
    user_diff_src_mem_->set_data_handle(gx);
    if (flags_ & use_scale_shift) {
        weights_mem_->set_data_handle(W);
        diff_weights_mem_->set_data_handle(gW);
    }
}

template<typename T>
int BatchNormalization<T>::backward(
    T* x, int x_d1, int x_d2, int x_d3, int x_d4,
    T* W, int W_d1, int W_d2,
    T* mean, int mean_d1, T* var, int var_d1,
    T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
    T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
    T* gW, int gW_d1, int gW_d2)
{
    if (!bwd_stream_) {
        backward_setup(
            x, x_d1, x_d2, x_d3, x_d4,
            W, W_d1, W_d2,
            mean, mean_d1, var, var_d1,
            gy, gy_d1, gy_d2, gy_d3, gy_d4,
            gx, gx_d1, gx_d2, gx_d3, gx_d4,
            gW, gW_d1, gW_d2);
        bwd_reset_mem(x, W, mean, var, gy, gx, gW);
        bwd_stream_->submit(bwd_primitives_).wait();
    }
    else {
        bwd_reset_mem(x, W, mean, var, gy, gx, gW);
        bwd_stream_->rerun().wait();
    }
    return 0;
}


template<typename T>
BatchNormalization<T>* BatchNormalization<T>::get_forward_object(
    int x_d1, int x_d2, int x_d3, int x_d4,
    int W_d1, int W_d2, int mean_d1, double eps,
    bool is_training, bool has_weights, bool fixed_mean_var)
{
    auto batch_normalization_forward = dynamic_cast<BatchNormalization<T>*>(
        LayerFactory<T>::get_instance().get_batch_normalization_layer(
            x_d1, x_d2, x_d3, x_d4, W_d1, W_d2, mean_d1, eps,
            is_training, has_weights, fixed_mean_var));

    if (batch_normalization_forward == NULL) {
        batch_normalization_forward = new BatchNormalization<T>(
            eps, is_training, has_weights, fixed_mean_var);
        LayerFactory<T>::get_instance().set_batch_normalization_layer(
            x_d1, x_d2, x_d3, x_d4, W_d1, W_d2, mean_d1, eps,
            is_training, has_weights, fixed_mean_var, batch_normalization_forward);
    }
    return batch_normalization_forward;
}

template<typename T>
BatchNormalization<T>* BatchNormalization<T>::get_backward_object(
    int x_d1, int x_d2, int x_d3, int x_d4,
    int W_d1, int W_d2, int mean_d1, double eps,
    bool is_training, bool has_weights, bool fixed_mean_var)
{
    auto batch_normalization_backward = dynamic_cast<BatchNormalization<T>*>(
        LayerFactory<T>::get_instance().get_batch_normalization_layer(
            x_d1, x_d2, x_d3, x_d4, W_d1, W_d2, mean_d1, eps,
            is_training, has_weights, fixed_mean_var));
    assert (batch_normalization_backward != NULL);  // already done forward before
    return batch_normalization_backward;
}


template class BatchNormalization<float>;
// template class BatchNormalization<double>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
