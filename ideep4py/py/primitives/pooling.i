%{
    #define SWIG_FILE_WITH_INIT
    #include "pooling_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"
%include "pooling_py.h"

%template(Pooling2D_Py_F32) Pooling2D_Py<float>;

//
// Python API for Pooling2D
//
// std::vector<mdarray*> Pooling2D_Py::Forward(
//                        mdarray *src,
//                        pooling_prarm_t *pp);
// mdarray* Pooling2D_Py::Backward(
//                        mdarray *diff_dst,
//                        mdarray *ws,
//                        conv_param_t *pp);
