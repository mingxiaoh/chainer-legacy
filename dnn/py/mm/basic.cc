#include "basic.h"
#include "tensor.h"

PyObject *basic::copyto(mdarray *dst, mdarray *src)
{
    Tensor *tdst = dst->get()->tensor();
    Tensor *tsrc = src->get()->tensor();
    if (tdst->copyto(tsrc) == true)
        Py_RETURN_NONE;
    return nullptr;
}

PyObject *basic::copyto(mdarray *dst, Py_buffer *src_view)
{
    // Validate it in ideepy code
    Tensor *tdst = dst->get()->tensor();
    if (tdst->len() != src_view->len) {
        return nullptr;
    }
    tdst->copyto((char *)src_view->buf);
    Py_RETURN_NONE;
}
