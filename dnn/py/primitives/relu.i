%{
    #define SWIG_FILE_WITH_INIT
    #include "relu_py.h"
%}

%include "std_vector.i"
%include "relu_py.h"

%template(Relu_Py_F32) Relu_Py<float>;
