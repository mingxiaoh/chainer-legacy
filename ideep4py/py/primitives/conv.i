%{
    #define SWIG_FILE_WITH_INIT
    #include "conv_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "conv_py.h"

%template(convolution2D) Convolution2D_Py<float>;

//
// Python API for Convolution2D
//
// mdarray Convolution2D_Py::Forward(
//                        mdarray *src, mdarray *weights, 
//                        mdarray *dst, mdarray *bias,
//                        conv_param_t *cp);
// std::vector<mdarray> Convolution2D_Py::BackwardWeights(
//                        mdarray *src, mdarray *diff_dst,
//                        con_prarm_t *cp);
// mdarray Convolution2D_Py::BackwardData(
//                        mdarray *weights, mdarray *diff_dst,
//                        conv_param_t *cp);
