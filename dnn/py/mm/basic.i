%{
  #include "basic.h"
%}

class basic {
public:
    static PyObject *copyto(mdarray *dst, mdarray *src);
    static PyObject *copyto(mdarray *dst, Py_buffer *view);
};

