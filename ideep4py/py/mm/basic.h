#pragma once
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL basic_ARRAY_API
#include <Python.h>
#include "mdarray.h"

class basic {
public:
    static PyObject *copyto(mdarray *dst, mdarray *src);
    static PyObject *copyto(mdarray *dst, Py_buffer *view);
    static mdarray acc_sum(vector<mdarray *> arrays);
};
