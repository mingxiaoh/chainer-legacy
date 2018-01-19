%{
    #define SWIG_FILE_WITH_INIT
    #include "concat_py.h"
%}

%include "std_vector.i"
%include "concat_py.h"

%template(concat) Concat_Py<float>;

//
// Python API for Concat
//
// mdarray Concat_Py::Forward(
//                        std::vector<mdarray> src,
//                        int axis);
// std::vector<mdarray> Concat_Py::Backward(
//                        mdarray *diff_dst,
//                        std::vector<int> offsets,
//                        int axis);
