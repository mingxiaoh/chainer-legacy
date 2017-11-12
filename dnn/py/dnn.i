%module dnn

%init %{
  import_array();
  implementation::g_init();
%}

%include "mdarray.i"
%include "relu.i"
%include "conv.i"
%include "pooling.i"
