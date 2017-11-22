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


#ifndef _LRN_BWD_FACTORY_
#define _LRN_BWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "lrn_bwd.h"

template <typename T>
class LocalResponseNormalizationBwdFactory : public OpFactory<T>
{
private:
    LocalResponseNormalizationBwdFactory() {}
    ~LocalResponseNormalizationBwdFactory() {}

public:
    static LocalResponseNormalizationBwd<T>* get(mkldnn::memory::dims src_d,
                                mkldnn::memory::dims dst_d,
                                mkldnn::memory::dims ws_d,
                                mkldnn::memory::data_type ws_dt,
                                int n, double k, double alpha, double beta,
                                mkldnn::algorithm alg_kind) {

        LocalResponseNormalizationBwd<T>* lrn_backward = NULL;

        //try to find a suitable one in pool
        lrn_backward = dynamic_cast<LocalResponseNormalizationBwd<T>*> (
            LocalResponseNormalizationBwdFactory<T>::get_instance().get_lrn_bwd( src_d, dst_d, ws_d, ws_dt, n, k, alpha, beta, alg_kind));

        if (lrn_backward == NULL) {
            LOG(INFO) << "create a new one for lrn bwd: " << alg_kind;
            lrn_backward = new LocalResponseNormalizationBwd<T>( src_d, dst_d, ws_d, ws_dt, n, k, alpha, beta, alg_kind);
            LocalResponseNormalizationBwdFactory<T>::get_instance().set_lrn_bwd( src_d, dst_d, ws_d, ws_dt, n, k, alpha, beta, alg_kind, lrn_backward);
        } else {
            LOG(INFO) << "reuse exist one for lrn bwd: " << alg_kind;
        }
        return lrn_backward;
    }

    static LocalResponseNormalizationBwdFactory& get_instance() {
        static LocalResponseNormalizationBwdFactory instance_;
        return instance_;
    }

private:
#define LRN_BWD_PREFIX "lrn_bwd_"
    Op<T>* get_lrn_bwd(mkldnn::memory::dims src_d,
                             mkldnn::memory::dims dst_d,
                             mkldnn::memory::dims ws_d,
                             mkldnn::memory::data_type ws_dt,
                             int n, double k, double alpha, double beta,
                             mkldnn::algorithm alg_kind) {
        std::string key = LRN_BWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += dims_to_string(ws_d);
        key += int_to_string(ws_dt);
        key += int_to_string(n);
        key += double_to_string(k);
        key += double_to_string(alpha);
        key += double_to_string(beta);
        key += int_to_string(alg_kind);

        return this->get_op(key);
    };

    void set_lrn_bwd(mkldnn::memory::dims src_d,
                           mkldnn::memory::dims dst_d,
                           mkldnn::memory::dims ws_d,
                           mkldnn::memory::data_type ws_dt,
                           int n, double k, double alpha, double beta,
                           mkldnn::algorithm alg_kind,
                           Op<T> *op) {
        std::string key = LRN_BWD_PREFIX;

        key += dims_to_string(src_d);
        key += dims_to_string(dst_d);
        key += dims_to_string(ws_d);
        key += int_to_string(ws_dt);
        key += int_to_string(n);
        key += double_to_string(k);
        key += double_to_string(alpha);
        key += double_to_string(beta);
        key += int_to_string(alg_kind);

        this->set_op(key, op);
    }
};

#endif // _LRN_BWD_FACTORY_