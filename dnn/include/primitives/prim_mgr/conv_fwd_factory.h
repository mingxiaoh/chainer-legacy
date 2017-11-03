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


#ifndef _CONV_FWD_FACTORY_
#define _CONV_FWD_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "op.h"
#include "op_factory.h"
#include <unordered_map>
#include "utils.h"
#include "conv_fwd.h"

template <typename T>
class Convolution2DFwdFactory : public OpFactory<T> 
{
private:
    Convolution2DFwdFactory();
    ~Convolution2DFwdFactory();

public:
    static Convolution2DFwd<T>* get( mkldnn::memory::dims x, mkldnn::memory::dims w,
                         mkldnn::memory::dims b, mkldnn::memory::dims y,
                         int sy, int sx,
                         int pad_lh, int pad_lw, int pad_rh, int pad_rw) {
        Convolution2DFwd<T>* conv2d_forward = NULL;

        //try to find a suitable one in pool
        conv2d_forward = dynamic_cast<Convolution2DFwd<T>*> (
                            Convolution2DFwdFactory<T>::get_instance().get_conv2d_fwd( x, w, b, y, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw));
        
        if (conv2d_forward == NULL) {
            LOG(INFO) << "create a new one for conv2d fwd";
            conv2d_forward = new Convolution2DFwd<T>( x, w, b, y, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw);
            Convolution2DFwdFactory<T>::get_instance().set_conv2d_fwd( x, w, b, y, sy, sx, pad_lh, pad_lw, pad_rh, pad_rw, conv2d_forward);
        } else {
            LOG(INFO) << "reuse exist one for conv2d fwd";
        }
        return conv2d_forward;
    }

    static Convolution2DFwdFactory& get_instance() {
        static Convolution2DFwdFactory instance_;
        return instance_;
    }

private:    
    Op<T>* get_conv2d_fwd( mkldnn::memory::dims x, mkldnn::memory::dims w,
                              mkldnn::memory::dims b, mkldnn::memory::dims y,
                              int sy, int sx, 
                              int pad_lh, int pad_lw, int pad_rh, int pad_rw);

    void set_conv2d_fwd( mkldnn::memory::dims x, mkldnn::memory::dims w,
                         mkldnn::memory::dims b, mkldnn::memory::dims y,
                         int sy, int sx,
                         int pad_lh, int pad_lw, int pad_rh, int pad_rw, 
                         Op<T>*     op);
};

#endif // _CONV_FWD_FACTORY_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
