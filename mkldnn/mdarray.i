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
%import memory.i

%buffer_protocol_producer(mdarray)
%buffer_protocol_typemap(Py_buffer *view)
%getattr_wrapper(mdarray)
%number_protocol(mdarray)

%immutable mdarray::dtype;
%immutable mdarray::size;
%immutable mdarray::ndim;

%extend mdarray {
  PyObject *dtype;
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

class mdarray {
public:
  static constexpr int MAX_NDIM = 12; //XXX: Same as MKLDNN
  typedef size_t size_type;

  mdarray(mkldnn::memory::dims dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &e);

  mdarray(mkldnn::memory::primitive_desc pd);

  mdarray(Py_buffer *view
      , mkldnn::memory::format, mkldnn::engine &);
};

%attribute_readonly(mdarray, mkldnn::memory *, memory, memory_get, mdarray::memory_get);
%attribute_readonly(mdarray, PyObject *, shape, shape_get, mdarray::shape_get);

using namespace mkldnn;

class s_op: public mdarray {
public:
  s_op (mkldnn::memory::primitive_desc dst
      , std::vector<mkldnn::primitive> *dag);
};

class d_op: public mdarray {
public:
  d_op(mkldnn::memory::primitive_desc gW
      , mkldnn::memory::primitive_desc gb
      , std::vector<mkldnn::primitive> *dag);
};

template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public s_op {
public:
  f_s_op(pd_t &op, mdarray &x, mdarray &W, mdarray &b
    , std::vector<primitive> *dag);
  f_s_op(pd_t &op, mdarray &x, mdarray &W
      , std::vector<primitive> *dag);
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public s_op {
public:
  bd_op(pd_t &op
      , mdarray &gy, mdarray &W, std::vector<primitive> *dag);
};

%typemap(out) s_op *extra %{
  $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $1_descriptor, 0 );
  PyObject_SetAttrString($result, "_ref", $self);
%}

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public d_op {
public:
  bwb_op(pd_t &op
      , mdarray &x, mdarray &gy, std::vector<primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public s_op {
public:
  bw_op(pd_t &op
      , mdarray &x, mdarray &gy, std::vector<primitive> *dag);
};

%attribute_readonly(bwb_op<convolution_backward_weights>
  , s_op *, extra, extra_get
  , bwb_op<convolution_backward_weights>::extra_get);

%template (conv_f_op) f_s_op<convolution_forward>;
%template (conv_bd_op) bd_op<convolution_backward_data>;
%template (conv_bwb_op) bwb_op<convolution_backward_weights>;
%template (conv_bw_op) bw_op<convolution_backward_weights>;
