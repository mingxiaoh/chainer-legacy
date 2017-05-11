%{
  template <class T>
  struct buffer_traits {
    static int getbuffer(PyObject *self, Py_buffer *view, int flags) {
      void *that;

      int res1 = SWIG_ConvertPtr(self, &that, nullptr, 0);

      if (!SWIG_IsOK(res1)) {
        PyErr_SetString(PyExc_ValueError, "Wrong self object in getbuffer wrapper");
        return -1;
      }

      // TODO: support smart pointer and raw at same time
      return (*reinterpret_cast<T *>(that))->getbuffer(self, view, flags);
    }

    static Py_ssize_t getreadbuf (PyObject *self, Py_ssize_t segment, void **ptrptr) { return 0; }
    static Py_ssize_t getwritebuf(PyObject *self, Py_ssize_t segment, void **ptrptr) { return 0; }
    static Py_ssize_t getcharbuf (PyObject *self, Py_ssize_t segment, void **ptrptr) { return 0; }
    static Py_ssize_t getsegcount(PyObject *self, Py_ssize_t *lenp) { return 0; }
  };
%}

%define %buffer_protocol_producer(type...)
  %feature("python:bf_getbuffer") type "buffer_traits<" %str(type) ">::getbuffer";
  %feature("python:bf_getreadbuffer") type "buffer_traits<" %str(type) ">::getreadbuf";
  %feature("python:bf_getwritebuffer") type "buffer_traits<" %str(type) ">::getwritebuf";
  %feature("python:bf_getsegcount") type "buffer_traits<" %str(type) ">::getsegcount";
  %feature("python:bf_getcharbuffer") type "buffer_traits<" %str(type) ">::getcharbuf";
%enddef

%define %buffer_protocol_typemap(VIEW)
%typemap(typecheck) (VIEW) {
  $1 = PyObject_CheckBuffer($input);
}

%typemap(in) (VIEW) (int res, Py_buffer *view = nullptr
  , int flags = PyBUF_C_CONTIGUOUS | PyBUF_RECORDS) {
  view = new Py_buffer;
  res = PyObject_GetBuffer($input, view, flags);
  $1 = ($1_ltype) view;
  // TODO: IF WE CONFRONT A F_CONTINGUOUS ONE???
}
%enddef
