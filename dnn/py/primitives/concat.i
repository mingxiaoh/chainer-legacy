%{
    #define SWIG_FILE_WITH_INIT
    #include "concat_py.h"
%}

%include "std_vector.i"
%include "concat_py.h"

%template(Concat_Py_F32) Concat_Py<float>;

//
// Python API for Concat
//
// mdarray Concat_Py::Forward(
//                        std::vector<mdarray> src,
//                        int axis); 
