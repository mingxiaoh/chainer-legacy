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
#include "mkldnn.hpp"
#include "relu_fwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
ReluFwd<T>::ReluFwd(mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt)
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    // create relu primitive
    if (relu_fwd_ == nullptr) {
        setup(src_d, src_fmt);
    }
}

template<typename T>
ReluFwd<T>::~ReluFwd()
{
}

template<typename T>
void ReluFwd<T>::setup(mkldnn::memory::dims src_d, mkldnn::memory::format src_fmt)
{
    LOG(INFO) << "Relu forward_setup";
    assert(src_d != nullptr);

    /* create memory descriptors for relu data w/ no specified format */
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T>(),
                                   src_fmt));
    src_mpd_.reset(new memory::primitive_desc(*src_md_, cpu_engine));
    /* create a relu*/
    const double negative_slope = 0.0;
    fwd_desc_.reset(new eltwise_forward::desc(prop_kind::forward, algorithm::eltwise_relu,
                                             *src_md_, negative_slope));

    fwd_pd_.reset(new eltwise_forward::primitive_desc(*fwd_desc_, cpu_engine));

    //store the expected memory format
    src_fmt_ = src_fmt;
    dst_fmt_ = static_cast<mkldnn::memory::format>(fwd_pd_.get()->dst_primitive_desc().desc().data.format);
    
    // create memory primitive based on dummy data
    src_mem_.reset(new memory(*src_mpd_, dummy));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), dummy));

    /* create relu primitive and add it to net */
    relu_fwd_.reset(new eltwise_forward(*fwd_pd_, *src_mem_, *dst_mem_));

    fwd_primitives_.push_back(*relu_fwd_);
    return;
}

template<typename T>
void ReluFwd<T>::execute(void* src, void* dst)
{
    LOG(INFO) << "Relu forward";

    src_mem_->set_data_handle(src);
    dst_mem_->set_data_handle(dst);
    fwd_stream_->submit(fwd_primitives_);
    
    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    dst_mem_->set_data_handle(dummy);
    
    return;
}

template class ReluFwd<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
