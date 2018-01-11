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


#pragma once

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "tensor.h"

typedef enum _eltwise_algorithm {
    ELTWISE_RELU = mkldnn::eltwise_relu,
    ELTWISE_TANH = mkldnn::eltwise_tanh,
    ELTWISE_ELU = mkldnn::eltwise_elu,
    ELTWISE_SQUARE = mkldnn::eltwise_square,
    ELTWISE_ABS = mkldnn::eltwise_abs,
    ELTWISE_SQRT = mkldnn::eltwise_sqrt,
    ELTWISE_LINEAR = mkldnn::eltwise_linear,
    ELTWISE_BOUNDED_RELU = mkldnn::eltwise_bounded_relu,
    ELTWISE_SOFT_RELU = mkldnn::eltwise_soft_relu,
    ELTWISE_LOGISTIC = mkldnn::eltwise_logistic,
} eltwise_algorithm_t;


static inline mkldnn::algorithm ideepy2mkldnn_eltwise_algorithm(eltwise_algorithm_t alg_kind) {
    return (mkldnn::algorithm)alg_kind;
}

template <typename...> class Eltwise;
template <typename T1, typename T2>
class Eltwise<T1, T2> : public Layer<T1>
{
public:
    Eltwise();
    ~Eltwise();
    
    /*
     * Eltwise Forward
     * params:
     * src: input, x
     * dst: output, y
     * y = max(x, 0)
     */
    static Tensor *Forward(Tensor *src, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta); 

    /*
     * Eltwise backward data
     * params:
     * src: input, x
     * diff_dst: input, gy
     * dst: output, gx
     * gx = gy*y
     */
    static Tensor *Backward(Tensor *src, Tensor *diff_dst, eltwise_algorithm_t alg_kind, T2 alpha, T2 beta);
};


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
