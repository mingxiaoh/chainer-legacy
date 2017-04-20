/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

%module (package="mkldnn") inner_product
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  using mkldnn::handle_traits;
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import memory.i

namespace mkldnn {

namespace c_api {
  %include c_api.i
}

%template (mkldnn_primitive_t_handle) handle< c_api::mkldnn_primitive_t >;
%template (mkldnn_engine_t_handle) handle< c_api::mkldnn_engine_t >;
%template (mkldnn_primitive_desc_t_handle) handle < c_api::mkldnn_primitive_desc_t >;
%template (mkldnn_stream_t_handle) handle< c_api::mkldnn_stream_t >;

%rename (f_desc) inner_product_forward::desc;
%rename (f_primitive_desc) inner_product_forward::primitive_desc;

%exception inner_product_forward::desc::desc {
  try {
    $action
  }
  catch (mkldnn::error &e){
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

struct inner_product_forward: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc);

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc);
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine);

        memory::primitive_desc src_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc bias_primitive_desc() const;

        memory::primitive_desc dst_primitive_desc() const;
    };

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const primitive::at &bias, const memory &dst);

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const memory &dst);
};

%rename (bd_desc) inner_product_backward_data::desc;
%rename (bd_primitive_desc) inner_product_backward_data::primitive_desc;

struct inner_product_backward_data: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc);
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc diff_dst_primitive_desc() const;

        memory::primitive_desc weights_primitive_desc() const;

        memory::primitive_desc diff_src_primitive_desc() const;
    };

    inner_product_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at weights,
            const memory &diff_src);
};

%rename (bw_desc) inner_product_backward_weights::desc;
%rename (bw_primitive_desc) inner_product_backward_weights::primitive_desc;

struct inner_product_backward_weights: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc);
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc);

        memory::primitive_desc diff_dst_primitive_desc() const;

        memory::primitive_desc diff_weights_primitive_desc() const;

        memory::primitive_desc diff_bias_primitive_desc() const;

        memory::primitive_desc src_primitive_desc() const;
    };

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights);

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights, const memory &diff_bias);
};

} // namespace mkldnn
