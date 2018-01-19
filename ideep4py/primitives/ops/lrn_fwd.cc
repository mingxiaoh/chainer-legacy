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
#include "lrn_fwd.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
LocalResponseNormalizationFwd<T>::LocalResponseNormalizationFwd(
    mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm)
    :alg_kind_(algorithm::lrn_across_channels)
{

    fwd_stream_.reset(new stream(stream::kind::eager));
    // setup
    if (fwd_ == NULL){
        setup(src_d, src_fmt, n, k, alpha, beta, alg_kind_);
    }
}

template<typename T>
LocalResponseNormalizationFwd<T>::~LocalResponseNormalizationFwd(){}

template<typename T>
void LocalResponseNormalizationFwd<T>::setup(
    mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm alg_kind)
{
    //LOG(INFO) << "lrn forward_setup";

    //LOG(INFO) << "src_d[0]=" << src_d[0] << "; src_d[1]" << src_d[1] << "; src_d[2]=" << src_d[2] << "; src_d[3]=" << src_d[3];
    alg_kind_ = alg_kind;
    // local_size_ = n;

    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
        get_desired_format(src_d[1]))); // use src's input channel to decide expected fmt
    // src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
    //                                src_fmt));

    //LOG(INFO) << "lrn_fwd_desc_";
    fwd_desc_.reset(new lrn_forward::desc(prop_kind::forward_training, alg_kind_, 
        *src_md_, n, alpha, beta, k));
    fwd_pd_.reset(new lrn_forward::primitive_desc(*fwd_desc_, cpu_engine));

    // store expected primitive format
    src_fmt_ = get_desired_format(src_d[1]);
    // src_fmt_ = src_fmt;
    //LOG(INFO) << "src_fmt is " << src_fmt <<" desired src_fmt_ is "<<src_fmt_;
    dst_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->dst_primitive_desc().desc().data.format);

    // create MKL-DNN internal memory object with dummy data
    src_mem_.reset(new memory({{{src_d}, memory_data_type<T>(), src_fmt_}, cpu_engine}, dummy));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), dummy));

    //need to return workspace for backward
    auto ws_pd = fwd_pd_.get()->workspace_primitive_desc().desc().data;
    // store workspace's dims and fmt to create ws tensor
    ws_fmt_ = static_cast<mkldnn::memory::format>(ws_pd.format);
    ws_dims_.assign(ws_pd.dims, ws_pd.dims + ws_pd.ndims);
    ws_dt_ = static_cast<mkldnn::memory::data_type>(ws_pd.data_type);
    ws_size_ = fwd_pd_.get()->workspace_primitive_desc().get_size();
    ws_mem_.reset(new memory(fwd_pd_.get()->workspace_primitive_desc(), dummy));

    fwd_.reset(new lrn_forward(
            *fwd_pd_, *src_mem_, *ws_mem_, *dst_mem_));
    
    fwd_primitives_.push_back(*fwd_);
    return;
}

template<typename T>
void LocalResponseNormalizationFwd<T>::execute(void *src, void *dst, void *ws)
{
    //LOG(INFO) << "lrn forward";
    
    src_mem_->set_data_handle(src); // input
    dst_mem_->set_data_handle(dst); // output dst

    assert(ws!=NULL);
    ws_mem_->set_data_handle(ws); // output workspace
        
    fwd_stream_->submit(fwd_primitives_);

    // set back data handle
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    
    assert(ws!=NULL);
    ws_mem_->set_data_handle(dummy);
    
    //LOG(INFO) << "lrn forward finish";
    return;
}

template class LocalResponseNormalizationFwd<float>;
