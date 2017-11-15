//NPY_NO_EXPORT PyObject *
// array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
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

  };
%}

%define %tp_richcompare(type...)
  %feature("python:tp_richcompare") type "tp_traits<" %str(type) ">::tp_richcompare";
%enddef
