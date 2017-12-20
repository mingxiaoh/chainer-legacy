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


#ifndef _POOLING_PY_H_
#define _POOLING_PY_H_

#include <vector>
#include <memory>
#include "op_param.h"
#include "mdarray.h"
#include "pooling.h"

template <typename T>
class Pooling2D_Py
{
public:
    /*
     * Python Pooling Forward
     * params:
     * src: input, x
     * pp: pooling parameters
     */
    static std::vector<mdarray> Forward(mdarray *src, 
                                        pooling_param_t *pp) {
        std::vector<mdarray> outputs;

        // Shoule be removed in future????
        implementation::mdarray *src_internal = src->get();
        
        std::vector<Tensor *> outputs_tensor = Pooling2D<T>::Forward(
                                                    (src_internal->tensor()),
                                                    pp);
        // FIXME
        //FIXME
        for (int i = 0; i < outputs_tensor.size(); i++) {
            outputs.push_back( mdarray(outputs_tensor[i]) );
        }

        return outputs;
    }

    /*
     * Python Pooling backward
     * param:
     * diff_dst: diff dst, gy
     * ws: workspace
     * pp: pooling parameters
     */
    static mdarray Backward(mdarray *diff_dst,
                            mdarray *ws,
                            pooling_param_t *pp) {
        //FIXME
        //Should be removed in future
        implementation::mdarray *diff_dst_internal = diff_dst->get();
        implementation::mdarray *ws_internal;
        if ( pp->algo_kind == pooling_param_t::algorithm::pooling_max)
            ws_internal = ws->get();
        
        Tensor *diff_src_tensor;
        if ( pp->algo_kind == pooling_param_t::algorithm::pooling_max) {
            diff_src_tensor = Pooling2D<T>::Backward(
                                    (diff_dst_internal->tensor()),
                                    (ws_internal->tensor()),
                                    pp);
        } else {
            diff_src_tensor = Pooling2D<T>::Backward(
                                    (diff_dst_internal->tensor()),
                                    NULL,
                                    pp);
        }

        // FIXME
        // In future, mdarray will have a Tensor member, no need to create a new one
        mdarray diff_src_mdarray = mdarray(diff_src_tensor);
        return diff_src_mdarray;
    }

};

#endif // _POOLING_PY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
