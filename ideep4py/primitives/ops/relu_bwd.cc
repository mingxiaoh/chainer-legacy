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
#include "relu_bwd.h"
#include "utils.h"
#include "common.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T1, typename T2>
EltwiseBwd<T1, T2>::EltwiseBwd(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format dst_diff_fmt, T2 alpha, T2 beta)
{
    bwd_stream_.reset(new stream(stream::kind::eager));
    // create eltwise primitive
    if (eltwise_bwd_ == nullptr) {
        setup(src_d, alg_kind, dst_diff_fmt, alpha, beta);
    }
}

template<typename T1, typename T2>
EltwiseBwd<T1, T2>::~EltwiseBwd()
{
}

template<typename T1, typename T2>
void EltwiseBwd<T1, T2>::setup(mkldnn::memory::dims src_d, mkldnn::algorithm alg_kind, mkldnn::memory::format dst_diff_fmt, T2 alpha, T2 beta)
{
    //LOG(INFO) << "Eltwise backward_setup";
    assert(src_d != nullptr);

    /* create memory descriptors for eltwise data w/ no specified format */
    src_md_.reset(new memory::desc({src_d}, memory_data_type<T1>(),
                                   dst_diff_fmt));
    dst_diff_md_.reset(new memory::desc({src_d}, memory_data_type<T1>(),
                                   dst_diff_fmt));
    src_mpd_.reset(new memory::primitive_desc(*src_md_, cpu_engine));
    dst_diff_mpd_.reset(new memory::primitive_desc(*dst_diff_md_, cpu_engine));
    /* create a eltwise*/
    fwd_desc_.reset(new eltwise_forward::desc(prop_kind::forward, alg_kind,
                                             *src_md_, alpha, beta));
    fwd_pd_.reset(new eltwise_forward::primitive_desc(*fwd_desc_, cpu_engine));

    bwd_desc_.reset(new eltwise_backward::desc(alg_kind,
                                               *dst_diff_md_, *src_md_, alpha, beta));

    bwd_pd_.reset(new eltwise_backward::primitive_desc(*bwd_desc_, cpu_engine, *fwd_pd_));

    //store the expected memory format
    src_diff_fmt_ = static_cast<mkldnn::memory::format>(bwd_pd_.get()->diff_src_primitive_desc().desc().data.format);
    
    // create memory primitive based on dummy data
    src_mem_.reset(new memory(*src_mpd_, dummy));
    dst_diff_mem_.reset(new memory(*dst_diff_mpd_, dummy));
    src_diff_mem_.reset(new memory(bwd_pd_.get()->diff_src_primitive_desc(), dummy));

    /* create eltwise primitive and add it to net */
    eltwise_bwd_.reset(new eltwise_backward(*bwd_pd_, *src_mem_, *dst_diff_mem_, *src_diff_mem_));

    bwd_primitives_.push_back(*eltwise_bwd_);
    return;
}

template<typename T1, typename T2>
void EltwiseBwd<T1, T2>::execute(void* src, void* dst_diff, void* src_diff)
{
    //LOG(INFO) << "Eltwise backward";

    src_mem_->set_data_handle(src);
    dst_diff_mem_->set_data_handle(dst_diff);
    src_diff_mem_->set_data_handle(src_diff);
    bwd_stream_->submit(bwd_primitives_);
    
    //after exec, set data handle back
    src_mem_->set_data_handle(dummy);
    dst_diff_mem_->set_data_handle(dummy);
    src_diff_mem_->set_data_handle(dummy);
    
    return;
}

template class EltwiseBwd<float, float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
