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
    if (tdst->len() != (size_t)src_view->len) {
        return nullptr;
    }
    tdst->copyto((char *)src_view->buf);
    Py_RETURN_NONE;
}

mdarray basic::acc_sum(vector<mdarray *> arrays)
{
    vector<shared_ptr<memory>> srcs_memory;
    vector<memory::primitive_desc> srcs_pd;
    vector<primitive::at> inputs;
    vector<float> scales;
    for (vector<mdarray *>::iterator it = arrays.begin();
            it != arrays.end(); it++) {
        Tensor *tensor = (*it)->get()->tensor();
        scales.push_back(1.0);
        srcs_pd.push_back(tensor->mkldnn_memory().get_primitive_desc());
        inputs.push_back(tensor->mkldnn_memory());
    }
    auto sum_pd = sum::primitive_desc(scales, srcs_pd);
    auto dst_pd = sum_pd.dst_primitive_desc();
    Tensor *dst_tensor = new Tensor(dst_pd);
    auto sum_p = sum(sum_pd, inputs, dst_tensor->mkldnn_memory());

    mkldnn::stream s(mkldnn::stream::eager);
    s.submit({sum_p}).wait();

    mdarray dst_mdarray = mdarray(dst_tensor);
    return dst_mdarray;
}
