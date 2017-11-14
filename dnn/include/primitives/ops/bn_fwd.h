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


#ifndef _BN_FWD_H_
#define _BN_FWD_H_

#include <mkldnn.hpp>
#include <vector>
#include "op.h"

template <typename T>
class batch_normalization_fwd : public Op<T> {
public:
    batch_normalization_fwd() : bn_fwd_(nullptr), src_mem_(nullptr),
                                flags_(0), pkind_(mkldnn::forward_training),
                                w_mem_(nullptr), dst_mem_(nullptr),
                                mean_mem_(nullptr), var_mem_(nullptr),
                                fwd_stream_(new mkldnn::stream(mkldnn::stream::kind::eager)) {}

    ~batch_normalization_fwd() {}

    void setup(mkldnn::memory::dims src_d, mkldnn::memory::dims w_d,
               float eps, bool global_stats, bool training);

    void execute(void *src, void *w, void *dst, void *mean, void *var);

public:
    std::shared_ptr<mkldnn::primitive> bn_fwd_;

private:
    unsigned long flags_;
    mkldnn::prop_kind pkind_;

    std::shared_ptr<mkldnn::memory> src_mem_;
    std::shared_ptr<mkldnn::memory> w_mem_;
    std::shared_ptr<mkldnn::memory> dst_mem_;
    std::shared_ptr<mkldnn::memory> mean_mem_;
    std::shared_ptr<mkldnn::memory> var_mem_;

    std::vector<mkldnn::primitive> fwd_primitives_;
    std::shared_ptr<mkldnn::stream> fwd_stream_;
};

#endif // _BN_FWD_H_
