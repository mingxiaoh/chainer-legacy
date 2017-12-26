%{
    #define SWIG_FILE_WITH_INIT
    #include "dropout_py.h"
    #include "op_param.h"
%}

%include "param.i"
%include "std_vector.i"
%include "dropout_py.h"

%template(dropout) Dropout_py<float>;
