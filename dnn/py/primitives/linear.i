%{
    #define SWIG_FILE_WITH_INIT
    #include "linear_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"
%include "linear_py.h"
%template(Linear_Py_F32) Linear_Py<float>;
%template(MdarrayVector) std::vector<mdarray>;
//
// API for Linear
// mdarray Linear_F32::Forward(
//              mdarray& src, mdarray& weights,
//              mdarray& dst, mdarray& bias,
//              linear_param_t& lp);
// std::vector<mdarray> Linear_F32::BackwardWeights(
//                              mdarray& src, mdarray& diff_dst,
//                              linear_param_t& lp);
// mdarray Linear_F32::BackwardData(
//                          mdarray& weights, mdarray& diff_dst,
//                          linear_param_t* lp);

