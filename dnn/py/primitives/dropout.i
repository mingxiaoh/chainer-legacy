%{
    #define SWIG_FILE_WITH_INIT
    #include "dropout_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"
%include "dropout_py.h"

%template(Dropout_F32) Dropout_py<float>;
