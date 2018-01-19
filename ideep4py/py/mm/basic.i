%{
  #include "basic.h"
%}

%typemap(in) (vector<mdarray *> arrays) {
    int i;
    int argc;
    vector<mdarray *> varr;
    if (!PyTuple_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expected a tuple");
        return nullptr;
    }
    argc = PyTuple_Size($input);
    for (i = 0; i < argc; i++) {
        PyObject *obj = PyTuple_GET_ITEM($input, i);
        if (!implementation::mdarray::is_mdarray(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expected a mdarray in acc_sum");
            return nullptr;
        }
#if 0
        if (!PyArray_Check(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expected a array");
            return nullptr;
        }
#endif
        void *that;
        int res1 = SWIG_ConvertPtr(obj, &that, nullptr, 0);
        if (!SWIG_IsOK(res1)) {
            PyErr_SetString(PyExc_ValueError, "Can't convert mdarray pyobject");
            return nullptr;
        }
        varr.push_back((mdarray *)that);
    }
    $1 = varr;
}

class basic {
public:
    static PyObject *copyto(mdarray *dst, mdarray *src);
    static PyObject *copyto(mdarray *dst, Py_buffer *view);
    static mdarray acc_sum(vector<mdarray *> arrays);
};

