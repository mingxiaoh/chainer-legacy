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


#ifndef _CONV_PY_H_
#define _CONV_PY_H_

#include <vector>
#include <memory>
#include "op_param.h"
#include "mdarray.h"
#include "conv.h"

template <typename T>
class Convolution2D_Py
{
public:
    /*
     * Python Convolution Forward
     * Y = W*X + b
     * params:
     * src: input, x
     * weight: weights, w
     * dst: output, y
     * bias: bias, b
     * cp: convolution parameters
     */
    static mdarray Forward(mdarray &src, 
                           mdarray &weights, 
                           mdarray &bias,
                           conv_param_t& cp) {
        // Shoule be removed in future????
        implementation::mdarray *src_internal = src.get();
        implementation::mdarray *w_internal = weights.get();
        implementation::mdarray *b_internal;
        if ( cp.with_bias )
            b_internal = bias.get();

        Tensor *dst_tensor = Convolution2D<T>::Forward(
                (*(src_internal->tensor())), 
                (*(w_internal->tensor())), 
                (*(b_internal->tensor())), cp);
        
        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray dst_mdarray = mdarray(dst_tensor);
        //mdarray dst_mdarray;
        return dst_mdarray;
    }

    /*
     * Python Convolution backward weights
     * gW = gy*x
     * params:
     * src: input, x
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static std::vector<mdarray> BackwardWeights(mdarray& src, 
                                                mdarray& diff_dst,
                                                conv_param_t& cp) {
        std::vector<mdarray> grads;

        //FIXME
        // Should be removed in future
        implementation::mdarray *src_internal = src.get();
        implementation::mdarray *diff_dst_internal = diff_dst.get();

        std::vector<Tensor *> grads_tensor = Convolution2D<T>::BackwardWeights(
                                        (*(src_internal->tensor())),
                                        (*(diff_dst_internal->tensor())),
                                        cp);
        
        //FIXME
        for (int i = 0; i < grads_tensor.size(); i++) {
            grads.push_back( mdarray(grads_tensor[i]) );
        }
        return grads;
    }

    /*
     * Python Convolution backward data
     * gx = gy*w
     * param:
     * weights: weights, w
     * diff_dst: diff dst, gy
     * cp: convolution parameters
     */
    static mdarray BackwardData(mdarray& weights, 
                                mdarray& diff_dst,
                                conv_param_t& cp) {
        //FIXME
        //Should be removed in future
        implementation::mdarray *w_internal = weights.get();
        implementation::mdarray *diff_dst_internal = diff_dst.get();

        Tensor *diff_src_tensor = Convolution2D<T>::BackwardData(
                                (*(w_internal->tensor())),
                                (*(diff_dst_internal->tensor())),
                                cp);

        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray diff_src_mdarray = mdarray(diff_src_tensor);
        return diff_src_mdarray;
    }

};

#endif // _CONV_PY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
