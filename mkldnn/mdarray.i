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
%import memory.i

%buffer_protocol_producer(mdarray)
%buffer_protocol_typemap(Py_buffer *view)

%immutable mdarray::memory;
%immutable mdarray::shape;
%immutable mdarray::dtype;
%immutable mdarray::size;
%immutable mdarray::ndim;

%extend mdarray {
  mkldnn::memory *memory;
  PyObject *shape;
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
