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

%extend mdarray {
  mkldnn::memory *memory;
  PyObject *shape;
  PyObject *dtype;
}

%{
  static PyObject *mdarray_shape_get(mdarray *self) {
    int ndim = self->ndims();
    PyObject *intTuple = PyTuple_New(ndim);
    auto m = self->memory();
    auto data = m.get_primitive_desc().desc().data;

    if (!intTuple)
      goto fail;

    for (int i = 0; i<ndim; i++) {
      PyObject *o = PyInt_FromLong((long)data.dims[i]); 

      if (!o) {
        Py_DECREF(intTuple);
        intTuple = NULL;
        goto fail;
      }

      PyTuple_SET_ITEM(intTuple, i, o);
    }

  fail:
    return intTuple;
  }

  mkldnn::memory *mdarray_memory_get(mdarray *self) {
    return &self->memory();
  }

  PyObject *mdarray_dtype_get(mdarray *self) {
    auto m = self->memory();

    PyArray_Descr *pd;
    // Translate our data_type to numpy one
    switch (m.get_primitive_desc().desc().data.data_type) {
      case mkldnn::memory::f32:
        pd = PyArray_DescrFromType(NPY_FLOAT);
        break;
      case mkldnn::memory::s32:
        pd= PyArray_DescrFromType(NPY_INT);
        break;
      default:
        return nullptr;
    }

    Py_INCREF(pd);
    return reinterpret_cast<PyObject *>(pd);
  }
%}

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

  void *data();
  size_type size();
};
