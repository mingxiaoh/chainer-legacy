%module dnn

%{
    #define SWIG_FILE_WITH_INIT
    #include "conv.h"
    #include "op_param.h"
%}

%include "mdarray.i"
%include "param.i"
%include "conv.h"

%init %{
  import_array();
  implementation::g_init();
%}

//
// API for Convolution2D
//
// mdarray* Convolution2D_F32::Forward(
//                        mdarray* src, mdarray* weights, 
//                        mdarray* dst, mdarray* bias,
//                        conv_param_t* cp);
// mdarray* Convolution2D_F32::BackwardWeights(
//                        mdarray* src, mdarray* diff_dst, mdarray* diff_bias,
//                        con_prarm_t cp);
// mdarray* Convolution2D_F32::BackwardData(
//                        mdarray* weights, mdarray* diff_dst,
//                        conv_param_t* cp);

%template(Convolution2D_F32) Convolution2D<float>;
