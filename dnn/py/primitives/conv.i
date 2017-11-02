%{
    #define SWIG_FILE_WITH_INIT
    #include "conv.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"

%template(MdarrayVector) std::vector<mdarray>;

//
// API for Convolution2D
//
// mdarray Convolution2D_F32::Forward(
//                        mdarray& src, mdarray& weights, 
//                        mdarray& dst, mdarray& bias,
//                        conv_param_t& cp);
// std::vector<mdarray> Convolution2D_F32::BackwardWeights(
//                        mdarray& src, mdarray& diff_dst,
//                        con_prarm_t cp);
// mdarray Convolution2D_F32::BackwardData(
//                        mdarray& weights, mdarray& diff_dst,
//                        conv_param_t* cp);

%include "conv.h"
%template(Convolution2D_F32) Convolution2D<float>;
