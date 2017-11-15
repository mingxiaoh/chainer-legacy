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
#include "relu_fwd.h"

template <typename T>
class ReluFwdFactory : public OpFactory<T>
{
private:
    ReluFwdFactory() {}
    ~ReluFwdFactory() {}

public:
    static ReluFwd<T>* get(mkldnn::memory::dims x, mkldnn::memory::format src_fmt) {
        ReluFwd<T>* relu_forward = nullptr;

        //try to find a suitable one in pool
        relu_forward = dynamic_cast<ReluFwd<T>*> (
                            ReluFwdFactory<T>::get_instance().get_relu_fwd(x, src_fmt));

        if (relu_forward == nullptr) {
            LOG(INFO) << "create a new one for relu fwd";
            relu_forward = new ReluFwd<T>(x, src_fmt);
            ReluFwdFactory<T>::get_instance().set_relu_fwd(x, src_fmt, relu_forward);
        } else {
            LOG(INFO) << "reuse exist one for relu fwd";
        }
        return relu_forward;
    }

    static ReluFwdFactory& get_instance() {
        static ReluFwdFactory instance_;
        return instance_;
    }

private:
#define RELU_FWD_PREFIX "relu_fwd_"
    Op<T>* get_relu_fwd(mkldnn::memory::dims x, mkldnn::memory::format src_fmt) {
        std::string key = RELU_FWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string(src_fmt);

        return this->get_op(key);
    }

    void set_relu_fwd(mkldnn::memory::dims x, mkldnn::memory::format src_fmt, Op<T>* op) {
        std::string key = RELU_FWD_PREFIX;

        key += dims_to_string(x);
        key += int_to_string(src_fmt);

        this->set_op(key, op);
    }
};
