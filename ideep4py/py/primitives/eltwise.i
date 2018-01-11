%{
    #define SWIG_FILE_WITH_INIT
    #include "eltwise_py.h"
%}

%include "std_vector.i"
%include "eltwise_py.h"

%template(relu) Relu_Py<float>;
%template(tanh) Tanh_Py<float>;
