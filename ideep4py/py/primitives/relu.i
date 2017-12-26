%{
    #define SWIG_FILE_WITH_INIT
    #include "relu_py.h"
%}

%include "std_vector.i"
%include "relu_py.h"

%template(relu) Relu_Py<float>;
