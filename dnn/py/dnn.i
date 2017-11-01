%module dnn

%init %{
  import_array();
  implementation::g_init();
%}

%include "mdarray.i"
%include "conv.i"
