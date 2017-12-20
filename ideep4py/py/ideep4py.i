%module ideep4py

%init %{
  import_array();
  implementation::g_init();
%}

%include "mdarray.i"
%include "relu.i"
%include "conv.i"
%include "pooling.i"
%include "linear.i"
%include "bn.i"
%include "concat.i"
%include "lrn.i"
%include "dropout.i"
