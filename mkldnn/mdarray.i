%module (package="mkldnn") mdarray
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include <numeric>
  #include <memory>
  #include <stdexcept>
  #include <mkldnn.hpp> 
#define SWIG_INLINE
  #include "mdarray.h"
%}

%init %{
  import_array();
  implementation::g_init();
%}

%include exception.i
%include pep_3118.i
%include getattro.i
%include asnumber.i
%include asmap.i
%include attribute.i

%import support.i
%import memory.i

%buffer_protocol_producer(mdarray)
%buffer_protocol_typemap(Py_buffer *view)
%getattr_wrapper(mdarray)
%number_protocol(mdarray)
%map_protocol(mdarray)

%define %codegen(Class, ret_type, attrib, getter)
%{
  inline ret_type %mangle(Class) ##_## attrib ## _get(Class *self_) {
    return (ret_type) Class::getter(self_);
  }
%}
%enddef

%define %extend_ro_attr(Class, ret_type, attrib, getter)
  %immutable Class::attrib;
  %extend Class {
    ret_type attrib;
  }
  %codegen(Class, ret_type, attrib, getter)
%enddef

%define %extend_ro_attr_and_own(Class, ret_type, attrib, getter)
  %immutable Class::attrib;
  %newobject Class::attrib;

  %extend Class {
    ret_type attrib;
  }

  %codegen(Class, ret_type *, attrib, getter)
%enddef

%extend_ro_attr(mdarray, PyObject *, dtype, mdarray_dtype_get)
%extend_ro_attr(mdarray, PyObject *, shape, mdarray_shape_get)
%extend_ro_attr(mdarray, long, size, mdarray_size_get)
%extend_ro_attr(mdarray, long, ndim, mdarray_ndim_get)
%extend_ro_attr_and_own(mdarray, mkldnn::memory, memory, mdarray_memory_get)

%{
  static int mdarray_setbuffer(mdarray *self, Py_buffer *view) {
    return (*self)->setbuffer(view);
  }
%}

%extend mdarray {
  int setbuffer(Py_buffer *view);
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
  // It is deliberately NOT matching prototypes!
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
class f_s_op: public mdarray {
public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
    , std::vector<mkldnn::primitive> *dag);
  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<mkldnn::primitive> *dag);
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public mdarray {
public:
  bd_op(pd_t &op, py_handle gy, py_handle W
  , std::vector<mkldnn::primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public mdarray {
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public mdarray {
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
  , std::vector<mkldnn::primitive> *dag);
};

// do not plan to create it from python interpreter

class reorder_buffer {
public:
  reorder_buffer(mdarray in);
};

