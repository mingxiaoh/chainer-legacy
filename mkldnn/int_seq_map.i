%define %int_sequence_map(integer_sequence_compitable_type)
%typemap(typecheck) (integer_sequence_compitable_type) {
  $1 = PySequence_Check($input);
}

%typemap(in) (integer_sequence_compitable_type) (int count) {
  count = PySequence_Size($input);

  for (int i =0; i < count; i ++) {
    PyObject *o = PySequence_GetItem($input, i);
    $1.push_back(PyLong_AsLong(o));
  }
}
%enddef
