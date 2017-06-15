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
#include "deconv.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
DeConvolution2D<T>::DeConvolution2D()
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    bwd_weights_stream_.reset(new stream(stream::kind::eager));
    bwd_data_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
DeConvolution2D<T>::~DeConvolution2D()
{
}

template<typename T>
void DeConvolution2D<T>::forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int pl1, int pl2,
        int pr1, int pr2)
{
    LOG(INFO) << "DeConvolution forward_setup";
    /*
    LOG(INFO) << "x =(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "W =(" << W_d1 << "," << W_d2 << "," << W_d3 << "," << W_d4 << ")";
    //LOG(INFO) << "b =(" << b_d1 << ")";
    LOG(INFO) << "y =(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";
    */
    
    /*
     * DeConvolution fwd --> Convolution bwd data
     */
    src_tz_ = {x_d1, x_d2, x_d3, x_d4};
    weights_tz_ = {W_d1, W_d2, W_d3, W_d4};
    dst_tz_ = {y_d1, y_d2, y_d3, y_d4};
    strides_ = {s1, s2};
    bias_tz_ = {b_d1};
    padding_l_ = {pl1, pl2};
    padding_r_ = {pr1, pr2};

    /* create memory for user data */
    user_src_mem_.reset(new memory({{{src_tz_}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine}, dummy));
    user_weights_mem_.reset(new memory({{{weights_tz_},
                                          memory_data_type<T>(), memory::format::oihw}, cpu_engine}, dummy));
    /* in current design, output is also allocated in python part */
    user_dst_mem_.reset(new memory({{{dst_tz_}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine}, dummy));

    /* create memory descriptors for convolution data w/ no specified format */
    src_md_.reset(new memory::desc({src_tz_}, memory_data_type<T>(),
                                   memory::format::any));
    weights_md_.reset(new memory::desc({weights_tz_},
                                       memory_data_type<T>(), memory::format::any));
    dst_md_.reset(new memory::desc({dst_tz_}, memory_data_type<T>(),
                                   memory::format::any));
    if ( b != NULL ) {
        bias_md_.reset(new memory::desc({bias_tz_}, memory_data_type<T>(),
                                    memory::format::any));
    }
    /* create a deconvolution */
    /*
     * convolution bwd data need convolution fwd as hint, so create convolution fwd here which will be used as deconvolution bwd data
     */
    fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                convolution_direct, *dst_md_, *weights_md_,
                *src_md_, strides_, padding_l_,padding_r_, padding_kind::zero));
    fwd_pd_.reset(new convolution_forward::primitive_desc(*fwd_desc_,cpu_engine));

    /*
     * use convolution bwd data to implement deconvolution fwd
     * data backward conv prim desc (gX = gy * W) ==> deconv: y = x * W
     * for data backward conv, no need b/gb
     * src_md: gx ==> deconv: y
     * weigths_md: W ==> deconv: W
     * dst_md: gy ==> deconv: x
     * */
    bwd_data_desc_.reset(new convolution_backward_data::desc(
                convolution_direct, *dst_md_, *weights_md_, 
                *src_md_, strides_, padding_l_, padding_r_, padding_kind::zero));
    bwd_data_pd_.reset(new convolution_backward_data::primitive_desc(
                *bwd_data_desc_, cpu_engine, *fwd_pd_));

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    // deconvolution fwd's x is gy in convolution bwd data 
    src_mem_ = user_src_mem_;
    if (memory::primitive_desc(bwd_data_pd_.get()->diff_dst_primitive_desc())
            != user_src_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "fwd reorder src dim";
        src_mem_.reset(new memory(bwd_data_pd_.get()->diff_dst_primitive_desc()));
        deconv_reorder_src_ = reorder(*user_src_mem_,*src_mem_);
        fwd_reorder_deconv_src_ = true;
    }

    // deconvolution fwd's w is w in convolution bwd data
    weights_mem_ = user_weights_mem_;
    if (memory::primitive_desc(bwd_data_pd_.get()->weights_primitive_desc())
        != user_weights_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "fwd reorder W";    
        weights_mem_.reset(new memory(bwd_data_pd_.get()->weights_primitive_desc()));
        deconv_reorder_weights_ = reorder(*user_weights_mem_, *weights_mem_);
        fwd_reorder_deconv_weights_ = true;
    } 

    // deconvolution fwd's y is gx in convolution bwd data
    dst_mem_ = user_dst_mem_;
    if (memory::primitive_desc(bwd_data_pd_.get()->diff_src_primitive_desc())
            != user_dst_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "fwd reorder y";
        dst_mem_.reset(new memory(bwd_data_pd_.get()->diff_src_primitive_desc()));
        deconv_reorder_dst_ = reorder(*dst_mem_, *user_dst_mem_);
        fwd_reorder_deconv_dst_ = true;
    }

    /* create deconvolution primitive and add it to net */
    deconv_bwd_data_.reset(new convolution_backward_data(
                *bwd_data_pd_, *src_mem_, *weights_mem_, *dst_mem_));

    //put all primitives into fwd_stream_
    if (fwd_reorder_deconv_src_){
        fwd_primitives_.push_back(deconv_reorder_src_);
    }
    if (fwd_reorder_deconv_weights_){
        fwd_primitives_.push_back(deconv_reorder_weights_);
    }
    fwd_primitives_.push_back(*deconv_bwd_data_);
    if (fwd_reorder_deconv_dst_){
        fwd_primitives_.push_back(deconv_reorder_dst_);
    }
    return;
}

template<typename T>
int DeConvolution2D<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int pl1, int pl2,
        int pr1, int pr2)
{
    LOG(INFO) << "DeConvolution forward" << b;
    if (deconv_bwd_data_ == NULL) {
        forward_setup(x, x_d1, x_d2, x_d3, x_d4,
                W, W_d1, W_d2, W_d3, W_d4,
                b, b_d1,
                y, y_d1, y_d2, y_d3, y_d4,
                s1, s2,
                pl1, pl2,
                pr1, pr2);
    }
    //LOG(INFO) << "conv_fwd_:" << conv_fwd_;
    //LOG(INFO) << "x=" << x << "; x_size=" << x_d1*x_d2*x_d3*x_d4*4;

    user_src_mem_->set_data_handle(x);
    user_weights_mem_->set_data_handle(W);
    user_dst_mem_->set_data_handle(y);
    if (fwd_first_run_) {
        fwd_stream_->submit(fwd_primitives_).wait();
        fwd_first_run_ = false;
    } else {
        fwd_stream_->rerun().wait();
    }

    return 0;
}

template<typename T>
int DeConvolution2D<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int pl1, int pl2,
        int pr1, int pr2)
{
    LOG(INFO) << "DeConvolution forward without bias";
//    LOG(INFO) << conv_fwd_;

    forward(x, x_d1, x_d2, x_d3, x_d4,
            W, W_d1, W_d2, W_d3, W_d4,
            NULL, -1,
            y, y_d1, y_d2, y_d3, y_d4,
            s1, s2,
            pl1, pl2,
            pr1, pr2);
    return 0;
}

template<typename T>
void DeConvolution2D<T>::backward_setup( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gb, int gb_d1)
{
    LOG(INFO) << "DeConvolution backward_setup";
#if 1
    /* create user format memory*/
    user_bwd_src_mem_.reset(new memory({{{ src_tz_ }, memory_data_type<T>(),
                memory::format::nchw }, cpu_engine }, dummy)); //x
    user_bwd_weights_mem_.reset(new memory({{{ weights_tz_ }, memory_data_type<T>(),
                memory::format::oihw }, cpu_engine }, dummy)); //W
    user_bwd_diff_dst_mem_.reset(new memory({{{ dst_tz_ }, memory_data_type<T>(),
                memory::format::nchw }, cpu_engine }, dummy)); //gy
    user_bwd_diff_weights_mem_.reset(new memory({{{ weights_tz_ }, memory_data_type<T>(),
                memory::format::oihw }, cpu_engine }, dummy)); //gW
    user_bwd_diff_src_mem_.reset(new memory({{{ src_tz_ }, memory_data_type<T>(),
                memory::format::nchw }, cpu_engine }, dummy)); //gx
    if ( b != NULL ) {
        user_bwd_diff_bias_mem_.reset(new memory({{{ bias_tz_}, memory_data_type<T>(),
                    memory::format::x,}, cpu_engine}, dummy)); //gB
    }
    /*
     * create backward deconvolution operator desc
     * memory desc can be shared between forward and backward, since they have same shape
     * */
    if ( b != NULL) {
        /*
         * weight backward deconv desc (gW = gy * X)
         * src_md: x == convolution_backward_weights
         * weigths_md: gW
         * bias_md: gb
         * dst_md: gy == convolution backward src
         * */
        bwd_weights_desc_.reset(new convolution_backward_weights::desc(
                    convolution_direct, *dst_md_, *weights_md_,
                    *bias_md_, *src_md_, strides_, padding_l_, padding_r_, padding_kind::zero));
    } else {
        /*
         * weight backward deconv prim desc (gW = gy * X)
         * src_md: x == convolution backward weights
         * weigths_md: gW
         * dst_md: gy == convolution backward src
         * */
        bwd_weights_desc_.reset(new convolution_backward_weights::desc(
                    convolution_direct, *dst_md_, *weights_md_,
                    *src_md_, strides_, padding_l_, padding_r_, padding_kind::zero));
    }

    /*
     * For deconv backward:
     * deconv backward weight == conv backward weight
     * deconv backward data == conv forward
     * */
    bwd_weights_pd_.reset(new convolution_backward_weights::primitive_desc(
                *bwd_weights_desc_, cpu_engine, *fwd_pd_));

    /*
     * for best performance convolution backward might choose different memory format for src and diffsrc
     * than the memory formats preferred by forward convolution for src and dst respectively
     * create reorder primitive for src from forward convolution to the format chosen by backward convolution */

    /* user_bwd_src_mem_ ==> x */
    bwd_src_mem_ = user_bwd_src_mem_;
    if (memory::primitive_desc(bwd_weights_pd_.get()->diff_dst_primitive_desc())
            != user_bwd_src_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd reorder x";
        bwd_src_mem_.reset(new memory(bwd_weights_pd_.get()->diff_dst_primitive_desc()));
        deconv_bwd_reorder_src_ = reorder(*user_bwd_src_mem_, *bwd_src_mem_);
        bwd_reorder_src_ = true;
    }

    /* user_bwd_diff_dst_weights_mem_ ==> gy for gW*/
    bwd_diff_dst_weights_mem_ = user_bwd_diff_dst_mem_;
    if (memory::primitive_desc(bwd_weights_pd_.get()->src_primitive_desc())
            != user_bwd_diff_dst_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd reorder gy";
        bwd_diff_dst_weights_mem_.reset(new memory(bwd_weights_pd_.get()->src_primitive_desc()));
        deconv_bwd_reorder_dst_weights_ = reorder(*user_bwd_diff_dst_mem_, *bwd_diff_dst_weights_mem_);
        bwd_reorder_diff_dst_weights_ = true;
    }

    /* user_bwd_diff_weights_mem_ ==> gW */
    bwd_diff_weights_mem_ = user_bwd_diff_weights_mem_;
    if (memory::primitive_desc(bwd_weights_pd_.get()->diff_weights_primitive_desc())
            != user_bwd_diff_weights_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd reorder gW";
        bwd_diff_weights_mem_.reset(new memory(bwd_weights_pd_.get()->diff_weights_primitive_desc()));
        deconv_bwd_reorder_diff_weights_ = reorder(*bwd_diff_weights_mem_, *user_bwd_diff_weights_mem_);
        bwd_reorder_diff_weights_ = true;
    }

    /* create weight deconv bwd prim */
    if (b != NULL) {
        /*
         * create deconvolution backward primitive (gW = gy * X)
         * src_mem: x ==> convolution backward diff dst
         * diff_dst_mem: gy ==> convolution backward src
         * diff_weights_mem: gW
         * diff_bias_mem: gb
         * */
        deconv_bwd_weights_.reset( new convolution_backward_weights(
                    *bwd_weights_pd_, *bwd_diff_dst_weights_mem_, *bwd_src_mem_,
                    *bwd_diff_weights_mem_, *user_bwd_diff_bias_mem_));
    } else {
        /*
         * create deconvolution backward primitive (gW = gy * x)
         * src_mem: x
         * diff_dst_mem: gy
         * diff_weights_mem: gW
         * */
        deconv_bwd_weights_.reset( new convolution_backward_weights(
                    *bwd_weights_pd_, *bwd_diff_dst_weights_mem_,
                    *bwd_src_mem_, *bwd_diff_weights_mem_));
    }

    /*
     * create weight deconv bwd stream (gW = gy * X)
     *
     * reorder_x -> reorder_gy -> weight_conv_bwd -> reoder_gW
     *
     */
    if (bwd_reorder_src_) {
        bwd_weights_primitives_.push_back(deconv_bwd_reorder_src_);
    }
    if (bwd_reorder_diff_dst_weights_) {
        bwd_weights_primitives_.push_back(deconv_bwd_reorder_dst_weights_);
    }
    bwd_weights_primitives_.push_back(*deconv_bwd_weights_);
    if (bwd_reorder_diff_weights_) {
        bwd_weights_primitives_.push_back(deconv_bwd_reorder_diff_weights_);
    }

    /*
     * DeConvolution backward data == conv fwd
     * conv fwd pd has already been created in forward_setup
     */
    fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                convolution_direct, *dst_md_, *weights_md_,
                *src_md_, strides_, padding_l_,padding_r_, padding_kind::zero));
    fwd_pd_.reset(new convolution_forward::primitive_desc(*fwd_desc_,cpu_engine));

    //reorder gy
    bwd_diff_dst_data_mem_ = user_bwd_diff_dst_mem_;
    if (memory::primitive_desc(fwd_pd_.get()->src_primitive_desc())
            != user_bwd_diff_dst_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd data reorder gy dim";
        bwd_diff_dst_data_mem_.reset(new memory(fwd_pd_.get()->src_primitive_desc()));
        deconv_bwd_reorder_dst_data_ = reorder(*user_bwd_diff_dst_mem_, *bwd_diff_dst_data_mem_);
        bwd_reorder_diff_dst_data_ = true;
    }

    //reorder W
    bwd_weights_mem_ = user_bwd_weights_mem_;
    if (memory::primitive_desc(fwd_pd_.get()->weights_primitive_desc())
            != user_bwd_weights_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd data reorder W dim";
        bwd_weights_mem_.reset(new memory(fwd_pd_.get()->weights_primitive_desc()));
        deconv_bwd_reorder_weights_ = reorder(*user_bwd_weights_mem_, *bwd_weights_mem_);
        bwd_reorder_weights_ = true;
    }

    //reorder gx
    bwd_diff_src_mem_ = user_bwd_diff_src_mem_;
    if (memory::primitive_desc(fwd_pd_.get()->dst_primitive_desc())
            != user_bwd_diff_src_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "bwd data reorder gx dim";
        bwd_diff_src_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc()));
        deconv_bwd_reorder_diff_src_ = reorder( *bwd_diff_src_mem_, *user_bwd_diff_src_mem_);
        bwd_reorder_diff_src_ = true;
    }

    //create conv fwd primitive for deconvolution backward data
    deconv_fwd_.reset(new convolution_forward(*fwd_pd_, *bwd_diff_dst_data_mem_,
                *bwd_weights_mem_, *bwd_diff_src_mem_));

    /*
     * create data conv bwd stream (gX = gy * W)
     *
     * reorder_W -> reorder_gy -> data_conv_bwd -> reorder_gX
     *
     */
    if (bwd_reorder_weights_) {
        bwd_data_primitives_.push_back(deconv_bwd_reorder_weights_);
    }
    if (bwd_reorder_diff_dst_data_) {
        bwd_data_primitives_.push_back(deconv_bwd_reorder_dst_data_);
    }
    bwd_data_primitives_.push_back(*deconv_fwd_);
    if (bwd_reorder_diff_src_) {
        bwd_data_primitives_.push_back(deconv_bwd_reorder_diff_src_);
    }
#endif
    return;
}

template<typename T>
int DeConvolution2D<T>::backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gb, int gb_d1,
    bool first_layer)
{
#if 1
    LOG(INFO) << "DeConvolution backward with bias " << b;
    if (deconv_bwd_weights_ == NULL) {
        backward_setup(x, x_d1, x_d2, x_d3, x_d4,
                W, W_d1, W_d2, W_d3, W_d4,
                b, b_d1,
                gy, gy_d1, gy_d2, gy_d3, gy_d4,
                gW, gW_d1, gW_d2, gW_d3, gW_d4,
                gx, gx_d1, gx_d2, gx_d3, gx_d4,
                gb, gb_d1);
    }

    user_bwd_src_mem_->set_data_handle(x); //x
    user_bwd_weights_mem_->set_data_handle(W); //W
    user_bwd_diff_src_mem_->set_data_handle(gx); //gx
    user_bwd_diff_weights_mem_->set_data_handle(gW); //gW
    user_bwd_diff_dst_mem_->set_data_handle(gy); //gy

    if (b!=NULL) {
        user_bwd_diff_bias_mem_->set_data_handle(gb); //gb
    }

    if (bwd_first_run_) {
        bwd_weights_stream_->submit(bwd_weights_primitives_).wait();
        if (!first_layer)//first layer will no need to do backward data
           bwd_data_stream_->submit(bwd_data_primitives_).wait();
        bwd_first_run_ = false;
    } else {
        bwd_weights_stream_->rerun().wait();
        if (!first_layer)
           bwd_data_stream_->rerun().wait();
    }
#endif
    return 0;
}

template<typename T>
int DeConvolution2D<T>::backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
    bool first_layer)
{
#if 1
    LOG(INFO) << "Convolution backward without bias";
    backward(x, x_d1, x_d2, x_d3, x_d4,
            W, W_d1, W_d2, W_d3, W_d4,
            NULL, -1,
            gy, gy_d1, gy_d2, gy_d3, gy_d4,
            gW, gW_d1, gW_d2, gW_d3, gW_d4,
            gx, gx_d1, gx_d2, gx_d3, gx_d4,
            NULL, -1,
        first_layer);
#endif
    return 0;
}

template class DeConvolution2D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
