%module (package="mkldnn") mdarray
%{
  #define SWIG_FILE_WITH_INIT
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #include <numpy/arrayobject.h>
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include <numeric>
  #include <memory>
  #include <stdexcept>
  #include <mkldnn.hpp> 
  #include "mdarray.h"
%}

%init %{
  // XXX: Do lazy init?
  import_array();
%}

%include exception.i
%include pep_3118.i
%include getattro.i
%include asnumber.i
%include attribute.i

%import support.i
%import memory.i

%import convolution_forward.i
%import convolution_backward_data.i
%import convolution_backward_weights.i

%buffer_protocol_producer(mdarray)
%buffer_protocol_typemap(Py_buffer *view)
%getattr_wrapper(mdarray)
%number_protocol(mdarray)

%immutable mdarray::dtype;
%immutable mdarray::size;
%immutable mdarray::ndim;
%immutable mdarray::shape;
%immutable mdarray::memory;
%newobject mdarray::memory;

%extend mdarray {
  PyObject *dtype;
  PyObject *shape;
  mkldnn::memory memory;
  long size;
  long ndim;
}

%exception mdarray::mdarray {
  try {
    $action
  } catch (mkldnn::error &e) {
    SWIG_exception(SWIG_ValueError, e.message.c_str());
  }
}

class mdarray: public py_handle {
public:
  mdarray(mkldnn::memory::dims dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &e);

  mdarray(mkldnn::memory::primitive_desc pd);

  mdarray(Py_buffer *view
      , mkldnn::memory::format, mkldnn::engine &);
};

template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public py_handle {
public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
    , std::vector<mkldnn::primitive> *dag);
  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<mkldnn::primitive> *dag);
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public py_handle {
public:
  bd_op(pd_t &op, py_handle gy, py_handle W
  , std::vector<mkldnn::primitive> *dag);
};

// No need of it
// %typemap(out) s_op *extra %{
//   $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $1_descriptor, 0 );
//   PyObject_SetAttrString($result, "_ref", $self);
// %}

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public py_handle {
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public py_handle {
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

%attribute_readonly(bwb_op<mkldnn::convolution_backward_weights>
  , mdarray, extra, extra_get
  , bwb_op<mkldnn::convolution_backward_weights>::extra_get);

%template (conv_f_op) f_s_op<mkldnn::convolution_forward>;
%template (conv_bd_op) bd_op<mkldnn::convolution_backward_data>;
%template (conv_bwb_op) bwb_op<mkldnn::convolution_backward_weights>;
%template (conv_bw_op) bw_op<mkldnn::convolution_backward_weights>;
