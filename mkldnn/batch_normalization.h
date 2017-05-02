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


#pragma once
#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_


#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class BatchNormalization : public Layer<T>
{
public:
    BatchNormalization(double eps, bool is_training, bool has_weights, bool fixed_mean_var);
    ~BatchNormalization() {}
    int forward() { return 0; }

    static void do_forward(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var,  int var_d1,
        double eps, bool is_training, bool has_weights, bool fixed_mean_var)
    {
        // LOG(INFO) << "do forward";
        auto forward_object = get_forward_object(
            x_d1, x_d2, x_d3, x_d4, W_d1, W_d2, mean_d1,
            eps, is_training, has_weights, fixed_mean_var);

        // LOG(INFO) << "forward";
        forward_object->forward(x, x_d1, x_d2, x_d3, x_d4,
                                y, y_d1, y_d2, y_d3, y_d4,
                                W, W_d1, W_d2,
                                mean, mean_d1, var, var_d1);
    }
    static void do_backward(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var, int var_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gW, int gW_d1, int gW_d2,
        double eps, bool is_training, bool has_weights, bool fixed_mean_var)
    {
        auto backward_object = get_backward_object(
            x_d1, x_d2, x_d3, x_d4, W_d1, W_d2, mean_d1,
            eps, is_training, has_weights, fixed_mean_var);

        backward_object->backward(x,  x_d1,  x_d2,  x_d3,  x_d4,
                                  W, W_d1, W_d2,
                                  mean, mean_d1,
                                  var, var_d1,
                                  gy, gy_d1, gy_d2, gy_d3, gy_d4,
                                  gx, gx_d1, gx_d2, gx_d3, gx_d4,
                                  gW, gW_d1, gW_d2);
    }

private:
    void forward(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var, int var_d1);

    void forward_setup(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int y_d1, int y_d2, int y_d3, int y_d4,
        int W_d1, int W_d2, int mean_d1, int var_d1);

    void fwd_reset_mem(T* x, T* y, T* W, T* mean, T* var);

    int backward(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var, int var_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gW, int gW_d1, int gW_d2);

    int backward_setup(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2,
        T* mean, int mean_d1, T* var, int var_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gW, int gW_d1, int gW_d2);

    void bwd_reset_mem(T* x, T* W, T* mean, T* var, T* gy, T* gx, T* gW);


protected:
    static BatchNormalization<T>* get_forward_object(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int W_d1, int W_d2,
        int mean_d1, double eps, bool is_training, bool has_weights, bool fixed_mean_var);

    static BatchNormalization<T>* get_backward_object(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int W_d1, int W_d2, int mean_d1, double eps,
        bool is_training, bool has_weights, bool fixed_mean_var);

private:
    double eps_ = 0.0;
    unsigned flags_ = 0;
    bool forward_first_use_ = true;
    mkldnn::prop_kind fwd_prop_kind_;

    //forward
    std::shared_ptr<mkldnn::memory> user_src_mem_, user_dst_mem_, src_mem_,
        dst_mem_, weights_mem_, mean_mem_, var_mem_;

    std::shared_ptr<mkldnn::batch_normalization_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::batch_normalization_forward::primitive_desc> fwd_pd_;
    std::shared_ptr<mkldnn::batch_normalization_forward> batch_normalization_fwd_;

    //backward
    std::shared_ptr<mkldnn::memory> bwd_user_src_mem_, bwd_src_mem_, diff_src_mem_,
        user_diff_src_mem_, diff_dst_mem_, user_diff_dst_mem_, diff_weights_mem_;

    std::shared_ptr<mkldnn::batch_normalization_backward::desc> bwd_desc_;
    std::shared_ptr<mkldnn::batch_normalization_backward::primitive_desc> bwd_pd_;
    std::shared_ptr<mkldnn::batch_normalization_backward> batch_normalization_bwd_;

    std::unique_ptr<mkldnn::primitive> reorder_src_, reorder_dst_, reorder_diff_src_,
        reorder_bwd_src_, reorder_diff_dst_;
    std::shared_ptr<mkldnn::stream> fwd_stream_, bwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_, bwd_primitives_;

    std::shared_ptr<mkldnn::engine> eng_;

};

#endif // _BATCH_NORMALIZATION_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
