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


template <typename T>
LocalResponseNormalizationFwdBwd<T>::LocalResponseNormalizationFwdBwd(
    mkldnn::memory::dims diff_src_d, 
    mkldnn::memory::dims diff_dst_d,
    mkldnn::memory::dims ws_d,
    mkldnn::memory::data_type ws_dt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm alg_kind):alg_kind(mkldnn::algorithm::lrn_across_channels)
{
    bwd_stream_.reset(new stream(stream::kind::eager));
    // setup
    if ( bwd_ == NULL){
        setup(diff_src_d, diff_dst_d, ws_d, ws_dt, n, k, alpha, beta, alg_kind);
    }
}

template <typename T>
LocalResponseNormalizationFwdBwd<T>::~LocalResponseNormalizationFwdBwd(){}

template <typename T>
void LocalResponseNormalizationFwdBwd<T>::setup(
    mkldnn::memory::dims diff_src_d, 
    mkldnn::memory::dims diff_dst_d,
    mkldnn::memory::dims ws_d,
    mkldnn::memory::data_type ws_dt,
    int n, double k, double alpha, double beta,
    mkldnn::algorithm alg_kind)
{
    LOG(INFO) << "lrn backward_setup";
    
    alg_kind_ = alg_kind;
    local_size_ = n;

    // create memory desc
    diff_src_md_.reset(new memory::desc({diff_src_d}, memory_data_type<T>(),
    memory::format::any)); //

    diff_dst_md_.reset(new memory::desc({diff_dst_d}, memory_data_type<T>(),
    get_desired_format(diff_dst_d[1]))); // use diff dst chanel to decide fmt

    //Need a forward hint to create backward, will be removed in future
    // create a lrn descriptor
    fwd_desc_.reset(new lrn_forward::desc(prop_kind::forward_training, alg_kind,
        *diff_src_md_, *diff_dst_md_,
        local_size, alpha, beta, k));
    fwd_pd_.reset(new lrn_forward::primitive_desc( *fwd_desc_, cpu_engine));
    bwd_desc_.reset(new lrn_backward::desc(alg_kind_,
        *diff_src_md_, *diff_dst_md_,
        local_size_, alpha, beta, k));
    bwd_pd_.reset(new lrn_backward::primitive_desc(*bwd_desc_, cpu_engine,
        *fwd_pd_));

    // store expected primitive format
    diff_src_fmt_ = static_cast<mkldnn::memory::format>(bwd_pd_.get()->diff_src_primitive_desc().desc().data.format);
    diff_dst_fmt_ = get_desired_format(diff_dst_d[1]);

    // create MKL-DNN internal memory object with dummy data
    diff_src_mem_.reset(new memory(bwd_pd_.get()->diff_src_primitive_desc(), dummy));
    diff_dst_mem_.reset(new memory({{{diff_dst_d}, memory_data_type<T>(), diff_dst_fmt_}, cpu_engine}, dummy));

    // store workspace's dims and fmt to create ws tensor
    ws_fmt_ = get_desired_format(ws_d[1]);
    ws_mem_.reset(new memory({{{ws_d}, ws_dt, ws_fmt_}, cpu_engine}, dummy)); // use ws dims's channel to decide format
    
    bwd_.reset(new pooling_backward(
            *bwd_pd_, *diff_dst_mem_, *ws_mem_, *diff_src_mem_));

    bwd_primitives_.push_back(*bwd_);
    return;
}

template<typename T>
void LocalResponseNormalizationFwdBwd<T>::execute(void *diff_src, void *diff_dst, void *ws)
{
    LOG(INFO) << "lrn backward";
    
    diff_src_mem_->set_data_handle(diff_src); // input
    diff_dst_mem_->set_data_handle(diff_dst); // output dst
   
    assert(ws!=NULL);
    ws_mem_->set_data_handle(ws); // output workspace

        
    bwd_stream_->submit(bwd_primitives_);

    // set back data handle
    diff_src_mem_->set_data_handle(dummy);
    diff_dst_mem_->set_data_handle(dummy);
    assert(ws!=NULL);
    ws_mem_->set_data_handle(dummy);
    
    LOG(INFO) << "lrn backward finish";
    return;
}

template class LocalResponseNormalizationFwdBwd<float>;