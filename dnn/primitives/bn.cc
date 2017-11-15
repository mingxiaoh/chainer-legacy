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


#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "tensor.h"
#include "bn.h"
#include "bn_fwd.h"
#include "prim_factory.h"
#include "reorder_op.h"

// using namespace mkldnn;

template<typename T>
std::vector<Tensor *> batch_normalization<T>::Forward(
    Tensor *src, Tensor *w, Tensor *mean, Tensor *var, float eps) {

    assert(memory_data_type<T>() == src.cxx_data_type());

    auto scale_shift = w ? true : false;
    auto global_stats = mean ? true : false;
    auto training = mean ? false : true;

    auto bn_fwd = batch_normalization_fwd_factory<T>::get(
            src->dims, eps, scale_shift, global_stats, training);

    void *src_data = src->data();
    void *src_itnl = nullptr;
    if (src->cxx_format() != bn_fwd->get_src_fmt()) {
        auto reorder = ReorderFactory<T>::get(
            src->dims(), src->cxx_format(), bn_fwd->get_src_fmt());
        src_itnl = new avx::byte[src->len()];
        reorder->execute(src_data, src_itnl);
        src_data = src_itnl;
    }

    auto dst = new Tensor(src->ndims(), src->dims(),
                          bn_fwd->get_dst_fmt(), src->type());
    mean = training ?
           new Tensor(bn_fwd->get_mean_ndims(), bn_fwd->get_mean_dims(),
                      bn_fwd->get_mean_fmt(), src->type()) : nullptr;
    var = training ?
          new Tensor(bn_fwd->get_var_ndims(), bn_fwd->get_var_dims(),
                     bn_fwd->get_var_fmt(), src->type()) : nullptr;

    bn_fwd->execute(src_data, (w ? w->data() : nullptr),
                    dst->data(), (mean ? mean->data() : nullptr),
                    (var ? var->data() : nullptr));

    std::vector<Tensor *> outs;
    outs.push_back(dst);
    if (training) {
        outs.push_back(mean);
        outs.push_back(var);
    }

    if (src_itnl)
        delete dynamic_cast<avx::byte *>(src_itnl);

    return outs;
}
