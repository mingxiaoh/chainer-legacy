%{
  template <class T>
  struct tp_traits {
    static PyObject *tp_richcompare(PyObject *self, PyObject *other, int cmp_op) {
      PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
                                            , NPY_ARRAY_ELEMENTSTRIDES, nullptr);
      if (surrogate == nullptr)
        return nullptr;

      PyObject *res = PyObject_RichCompare(surrogate, other, cmp_op);
      Py_DECREF(surrogate);
      return res;
    }

    static PyObject *tp_iter(PyObject *self) {
      PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
                                            , NPY_ARRAY_ELEMENTSTRIDES, nullptr);
      if (surrogate == nullptr)
          return nullptr;

      PyObject *res = PyObject_GetIter(surrogate);
      Py_DECREF(surrogate);
      return res;
    }
  };
%}

%define %tp_slot(name, type)
  %feature("python:tp_" %str(name)) type "tp_traits<" %str(type) ">::tp_" %str(name);
%enddef

%define %tp_protocol(type...)
  %tp_slot(richcompare, type)
  %tp_slot(iter, type)
%enddef
