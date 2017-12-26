%{
    #define SWIG_FILE_WITH_INIT
    #include "bn_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"
%include "bn_py.h"

%template(batchNormalization) batch_normalization_py<float>;
