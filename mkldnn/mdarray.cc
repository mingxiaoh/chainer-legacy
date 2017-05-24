#include "mdarray.h"

namespace implementation {

static PyObject *PyType_reorder_buffer = nullptr;

static swig_type_info *SwigTy_mdarray = nullptr;
static swig_type_info *SwigTy_engine = nullptr;
static PyObject *PyType_mdarray = nullptr;

PyObject *queryPyTypeObject(const char *name) {
  swig_type_info *info = SWIG_TypeQuery(name);
  if (info != nullptr) {
    SwigPyClientData *cd
      = (SwigPyClientData *)info->clientdata;
    return reinterpret_cast<PyObject *>(cd->pytype);
  }

  throw mkldnn::error(mkldnn::c_api::mkldnn_invalid_arguments
      , "Failed to find reorder_buffer object");
}

// We brought this to global scope to mitigate it consumption
#if PY_VERSION_HEX >= 0x03000000
int g_init() {
#else
void g_init() {
#endif
  PyType_reorder_buffer = queryPyTypeObject("_p_reorder_buffer");
  SwigTy_mdarray = SWIG_TypeQuery("_p_mdarray");
  PyType_mdarray = queryPyTypeObject("_p_mdarray");
  SwigTy_engine = SWIG_TypeQuery("_p_mkldnn__engine");

  // XXX: I don't quite understand it, and its repercussions :)
  SwigPyObject_stype = SWIG_MangledTypeQuery("_p_SwigPyObject");

  if (SwigPyObject_stype == nullptr)
    throw mkldnn::error(mkldnn::c_api::mkldnn_invalid_arguments
        , "Failed to find SwigPyObject object");

  // Initiate static variables imported from numpy include
  import_array();

#if PY_VERSION_HEX >= 0x03000000
  return 0;
#else
  return;
#endif
}

// Pin the virtual table
PyArrayInterface *mdarray::getastr(reorder_buffer *rb) {
  rb->fire(this);
  return rb->build_astr();
}

//FIXME: macro SWIG_as_voidptr is copied from mdarray_wrap.cpp
#define SWIG_as_voidptr(a) const_cast< void * >(static_cast< const void * >(a)) 

PyObject *mdarray::m_Add(PyObject *self, PyObject *o) {
  // Resource manager
  auto py_decref = [](PyObject *p) { Py_DECREF(p); };

  std::unique_ptr<PyObject
    , decltype(py_decref)> op(nullptr);

  // Create mdarray from buffer provider
  if (reinterpret_cast<PyObject *>(o->ob_type) != PyType_mdarray) {
    mkldnn::engine p_e = get_engine();

    PyObject *Py_p_engine = SWIG_Python_NewPointerObj(nullptr
        , SWIG_as_voidptr(&p_e), SwigTy_engine, 0);

    PyObject *argList = Py_BuildValue("(OiO)", o, mkldnn::memory::nchw
        , Py_p_engine);

    if (argList == nullptr) {
      PyErr_SetString(PyExc_SystemError, "Can not create argument list");
      return nullptr;
    }

    o = PyObject_CallObject(PyType_mdarray, argList);

    Py_DECREF(argList);
    Py_DECREF(Py_p_engine);

    if (o == nullptr) {
      PyErr_SetString(PyExc_BufferError, "Cannot create mdarray from input");
      return nullptr;
    }

    op.reset(o);
  }

  void *oprd2;
  int res = SWIG_ConvertPtr(o, &oprd2, nullptr, 0);

  if (!SWIG_IsOK(res)) {
    PyErr_SetString(PyExc_ValueError, "Wrong operand object in add wrapper");
    return nullptr;
  }

  // 2 mdarray add
  auto mdarray2 = (reinterpret_cast<py_handle *>(oprd2))->get();

  mkldnn::sum::primitive_desc sum_pd({1.0, 1.0}
      , {m_.get_primitive_desc(), mdarray2->m_.get_primitive_desc()});

  std::vector<mkldnn::memory::primitive::at> inputs_at {memory()
    , mdarray2->memory()};

  py_handle *output = new py_handle(new mdarray(sum_pd.dst_primitive_desc()));

  mkldnn::sum sum_prim(sum_pd
      , inputs_at, (*output)->memory());

  mkldnn::stream s(mkldnn::stream::kind::eager);
  s.submit({sum_prim}).wait();

  PyObject *resultobj = SWIG_Python_NewPointerObj(nullptr
      , SWIG_as_voidptr(output), SwigTy_mdarray, SWIG_POINTER_OWN |  0 );

  return resultobj;
}

int mdarray::getbuffer(PyObject *self, Py_buffer *view, int flags) {
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
    PyErr_SetString(PyExc_ValueError, "carray is not Fortran contiguous");
    return -1;
  }

  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  // Reorder_buffer type object
  if (PyType_reorder_buffer == nullptr) {
    PyErr_SetString(PyExc_NameError, "name 'reorder_buffer' is not defined");
    return -1;
  }

  // Wrote some python in C++ :)
  PyObject *argList = Py_BuildValue("(O)", self);
  if (argList == nullptr) {
    return -1;
  }

  // TODO: Do we need to cache this thing?
  PyObject *rbobj = PyObject_CallObject(PyType_reorder_buffer, argList);
  Py_DECREF(argList);

  if (rbobj == nullptr) {
    return -1;
  }

  reorder_buffer *rb;
  SWIG_ConvertPtr(rbobj, reinterpret_cast<void **>(&rb), nullptr, 0);

  rb->fire(this);

  if ( rb->build_view(view, flags) ) {
    PyErr_SetString(PyExc_RuntimeError, "Can't build Py_buffer!");
    return -1;
  }

  // Stolen reference
  view->obj = rbobj;

  return 0;
}

PyObject *mdarray::getattro(PyObject *self, PyObject *name) {
  // XXX: Recursive alarm !!! XXX
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return nullptr;

  // Watch the reference count of surrogate if more compicated
  // looking up method involved
  PyObject * attr = PyObject_GetAttr(surrogate, name);

  // The surrogate will be destroyed after attribute is done
  Py_DECREF(surrogate);

  if (attr == nullptr && PyErr_ExceptionMatches(PyExc_AttributeError)) {
    PyErr_Clear();

    // Switch to our exception message if things gone wrong
    PyTypeObject *tp = Py_TYPE(self);
    PyErr_Format(PyExc_AttributeError
        , "'%.50s' object has no attribute '%U'", tp->tp_name, name);
  }

  return attr;
}

Py_ssize_t mdarray::mp_length(PyObject *self) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return -1;

  Py_ssize_t len = PyMapping_Length(surrogate);
  Py_DECREF(surrogate);

  // TODO: Exception localize
  return len;
}

PyObject *mdarray::mp_subscript(PyObject *self, PyObject *op) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  if (surrogate == nullptr)
    return nullptr;

  PyObject *ret = PyObject_GetItem(surrogate, op);
  Py_DECREF(surrogate);

  // TODO: Exception localize
  return ret;
}

int mdarray::mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op) {
  PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

  int ret;

  if (surrogate == nullptr)
    ret = -1;

  if (op == nullptr)
    ret = PyObject_DelItem(surrogate, ind);
  else
    ret = PyObject_SetItem(surrogate, ind, op);

  Py_DECREF(surrogate);

  // TODO: Exception localize
  return ret;
}

int s_op::getbuffer(PyObject *self, Py_buffer *view, int flags) {
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
    PyErr_SetString(PyExc_ValueError, "carray is not Fortran contiguous");
    return -1;
  }

  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  // Only for the first, framework do it for us next time
  if (reorder_ == nullptr) {
    reorder_.reset(new reorder_buffer(this));
    mkldnn::reorder rb_p = reorder_->fire(this);
    dag_->push_back(rb_p);
  }

  if ( reorder_->build_view(view, flags) ) {
    PyErr_SetString(PyExc_RuntimeError, "Can't build Py_buffer!");
    return -1;
  }

  view->obj = self;
  Py_INCREF(self);

  return 0;
}

}
