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

#pragma once

#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "relu_bwd.h"

template <typename T>
class ReluBwdFactory : public OpFactory<T>
{
private:
    ReluBwdFactory();
    ~ReluBwdFactory();

public:
    static ReluBwd<T>* get(mkldnn::memory::dims x, mkldnn::memory::format dst_diff_fmt) {
        ReluBwd<T>* relu_backward = nullptr;

        //try to find a suitable one in pool
        relu_backward = dynamic_cast<ReluBwd<T>*> (
                            ReluBwdFactory<T>::get_instance().get_relu_bwd(x, dst_diff_fmt));

        if (relu_backward == nullptr) {
            LOG(INFO) << "create a new one for relu bwd";
            relu_backward = new ReluBwd<T>(x, dst_diff_fmt);
            ReluBwdFactory<T>::get_instance().set_relu_bwd(x, dst_diff_fmt, relu_backward);
        } else {
            LOG(INFO) << "reuse exist one for relu bwd";
        }
        return relu_backward;
    }

    static ReluBwdFactory& get_instance() {
        static ReluBwdFactory instance_;
        return instance_;
    }

private:
#define RELU_BWD_PREFIX "relu_bwd_"
    Op<T>* get_relu_bwd(mkldnn::memory::dims x, mkldnn::memory::format dst_diff_fmt) {
        std::string key = RELU_BWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string(dst_diff_fmt);

        return this->get_op(key);
    }

    void set_relu_bwd(mkldnn::memory::dims x, mkldnn::memory::format dst_diff_fmt, Op<T> *op) {
        std::string key = RELU_BWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string(dst_diff_fmt);

        this->set_op(key, op);
    }
};


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
