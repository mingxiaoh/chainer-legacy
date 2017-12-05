%{
  #define SWIG_FILE_WITH_INIT
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include <numeric>
  #include <memory>
  #include <stdexcept>
  #include <stdarg.h>
#define SWIG_INLINE
  #include "mdarray.h"
%}

%include exception.i
%include pep_3118.i
%include getattro.i
%include asnumber.i
%include asmap.i
%include attribute.i
%include tp.i
%include std_vector.i

%template(MdarrayVector) std::vector<mdarray>;
%template(IntVector) std::vector<int>;

%tp_richcompare(mdarray)
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

/* mdarray::reshape */
%extend mdarray {
  %typemap(in) (...)(vector<int> args) {
     int i;
     int argc;
     argc = PyTuple_Size(varargs);
     if (argc > 4) {
       PyErr_SetString(PyExc_ValueError,"Too many arguments");
       return NULL;
     }
     if (argc == 1) {
       // args should be tuple
       PyObject *o = PyTuple_GetItem(varargs,0);
       if (!PyTuple_Check(o)) {
         PyErr_SetString(PyExc_ValueError,"Expected a tuple");
         return NULL;
       }
       Py_ssize_t size = PyTuple_GET_SIZE(o);
       if (size != 4 && size != 2) {
         PyObject *surrogate = PyArray_FromAny($self, nullptr, 0, 0
                 , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

         if (surrogate == nullptr) {
           PyErr_SetString(PyExc_ValueError,"Unexpected array");
           return nullptr;
         }
         PyObject *res = PyArray_Reshape((PyArrayObject *)surrogate, o);

         Py_DECREF(surrogate);
         return res;
       }
       for (i = 0; i < PyTuple_GET_SIZE(o); i++) {
         PyObject *obj = PyTuple_GET_ITEM(o, i);
         if (!PyInt_Check(obj) && !PyLong_Check(obj)) {
           PyErr_SetString(PyExc_ValueError,"Expected a int or long in tuple");
           return NULL;
         }
         args.push_back(PyInt_AsLong(obj));
       }
     } else {
       Py_ssize_t size = argc;
       if (size != 4 && size != 2) {
         PyObject *surrogate = PyArray_FromAny($self, nullptr, 0, 0
                 , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

         if (surrogate == nullptr) {
           PyErr_SetString(PyExc_ValueError,"Unexpected array");
           return nullptr;
         }
         PyObject *res = PyArray_Reshape((PyArrayObject *)surrogate, varargs);

         Py_DECREF(surrogate);
         return res;
       }
       for (i = 0; i < argc; i++) {
         PyObject *o = PyTuple_GetItem(varargs,i);
         if (!PyInt_Check(o) && !PyLong_Check(o)) {
           PyErr_SetString(PyExc_ValueError,"Expected a int");
           return NULL;
         }
         //args[i] = PyInt_AsLong(o);
         args.push_back(PyInt_AsLong(o));
       }
     }
     $1 = &args;
  }

  PyObject *reshape(...) {
    va_list vl;
    va_start(vl, self);
    vector<int> *dims = va_arg(vl, vector<int>*);
    va_end(vl);
    return (*self)->reshape(self, *dims);
  }
}

/* mdarray::sum */
%extend mdarray {
  %feature ("kwargs") sum;
  %typemap(in) vector<int> axis {
    $1.clear();
    if (PyTuple_Check(obj1)) {
      for (int i = 0; i < PyTuple_Size(obj1); i++) {
        PyObject *item = PyTuple_GetItem(obj1, i);
#if PY_VERSION_HEX > 0x03000000
        if (!PyLong_Check(item)) {
#else
        if (!PyInt_Check(item)) {
#endif
          SWIG_exception_fail(SWIG_ValueError,
              "in method '" "mdarray_sum" "', argument " "2"" of type '" "tuple (int, int, ...)""'");
          SWIG_fail;
        }

        $1.push_back(PyLong_AsLong(item));
      }
    } else if (PyLong_Check(obj1)) {
      $1.push_back(PyLong_AsLong(obj1));
    } else {
      void *_obj1;
      if (!SWIG_IsOK(SWIG_ConvertPtr(obj1, &_obj1, nullptr, 0))) {
        PyErr_SetString(PyExc_ValueError, "Wrong object in sum wrapper");
        SWIG_fail;
      }

      if (!_obj1) {
        $1.clear();
      } else {
        SWIG_exception_fail(SWIG_ValueError,
            "in method '" "mdarray_sum" "', argument " "2"" of type '" "tuple or int""'");
        SWIG_fail;
      }
    }
  }

  %typemap(argout) (vector<int> axis) {
    if (!$result) {
      auto *surrogate = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(
                            $self, nullptr, 0, 0, NPY_ARRAY_ELEMENTSTRIDES, nullptr));
      if (surrogate == nullptr)
        return nullptr;

      if (!$1.size()) {
        for (int i = 0; i < PyArray_NDIM(surrogate); i++)
          $1.push_back(i);
      }

      auto *res = surrogate;
      for (int i = 0; i < static_cast<int>($1.size()); i++) {
        auto *tmp = reinterpret_cast<PyArrayObject *>(PyArray_Sum(
                      res, $1[i], PyArray_TYPE(res), nullptr));
        for (int j = i + 1; j < $1.size(); j++) {
          if ($1[i] < $1[j])
            $1[j] -= 1;
        }

        // if (i < axis.size() - 1)
        //   Py_DECREF(res);

        Py_DECREF(res);
        res = tmp;
      }

      return reinterpret_cast<PyObject *>(res);
    }
  }

  PyObject *sum(vector<int> axis={0}, int dtype=0,
                PyObject *out=nullptr, bool keepdims=false) {
    return (*self)->sum(axis);
  }
}

/*
%extend mdarray {
  PyObject *__getstate__() {
    return (*$self)->__getstate__();
  }

  //TODO
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
*/

class mdarray: public py_handle {
public:
  // It is deliberately NOT matching prototypes!
  // FIXME
  // add default constructor so that native can pass vector<mdarray> to python
  mdarray();
  mdarray(Py_buffer *view, char input_type = 'd');
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

class reorder_buffer {
public:
  reorder_buffer(mdarray in);
};
