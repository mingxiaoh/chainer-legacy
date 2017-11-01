%{
  #define SWIG_FILE_WITH_INIT
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include <numeric>
  #include <memory>
  #include <stdexcept>
#define SWIG_INLINE
  #include "mdarray.h"
%}

%include exception.i
%include pep_3118.i
%include getattro.i
%include asnumber.i
%include asmap.i
%include attribute.i

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
%extend_ro_attr(mdarray, bool, is_mdarray, mdarray_is_mdarray_get)

%extend mdarray {
  int setbuffer(Py_buffer *view) {
    return (*$self)->setbuffer(view);
  }

  void reset_buf_order() {
    (*$self)->reset_buf_order();
  }

  PyObject *axpby(double a, double b, PyObject *y) {
    return (*$self)->axpby(a, b, y);
  }

  PyObject *inplace_axpby(double a, double b, PyObject *y) {
    /// Second param y is a harmless dummy
    return (*$self)->inplace_axpby(a, y, b, y);
  }

  PyObject *flat() {
    return (*self)->flat();
  }
}

%extend mdarray {
  PyObject *__getstate__() {
    return (*$self)->__getstate__();
  }

  %typemap(default) (PyObject *state) {
    PyObject *state;

    if (!PyArg_UnpackTuple(args, (char *)"mdarray___setstate__", 0, 1, &state)) SWIG_fail;

    if (!PyTuple_Check(state)) SWIG_fail;

    PyObject *py_dims = PyTuple_GetItem(state, 0);
    PyObject *py_dtype = PyTuple_GetItem(state, 1);
    PyObject *py_format = PyTuple_GetItem(state, 2);
    PyObject *py_engine = PyTuple_GetItem(state, 3);
    PyObject *py_rdata = PyTuple_GetItem(state, 4);

    void *rdata = PyLong_AsVoidPtr(py_rdata);

    mdarray *unpickled_mdarr = nullptr; //new mdarray(dims, dtype, format, engine);
    (*unpickled_mdarr)->unpickled_data(rdata);
    SwigPyObject *sobj = SWIG_Python_GetSwigThis(self);
    if (sobj) {
      sobj->ptr = reinterpret_cast<void *>(unpickled_mdarr);
      sobj->ty = SWIGTYPE_p_mdarray;
      sobj->own = 0;
      sobj->next = 0;
    } else {
      SWIG_fail;
    }
  }

  void __setstate__(PyObject *state) {
    (*$self)->__setstate__(state);
  }
}

class mdarray: public py_handle {
public:
  // It is deliberately NOT matching prototypes!
  mdarray(Py_buffer *view);
  virtual ~mdarray();
};

%typemap(in) (mdarray *in_mdarray) {
    void *that;
    int res1 = SWIG_ConvertPtr($input, &that, nullptr, 0);
    if (!SWIG_IsOK(res1)) {
        PyErr_SetString(PyExc_ValueError, "Can't convert mdarray pyobject");
        return nullptr;
    }
    $1 = (reinterpret_cast<mdarray *>(that));
};
