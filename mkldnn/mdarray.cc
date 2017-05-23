#include "mdarray.h"

namespace implementation {

static PyObject *PyType_reorder_buffer = nullptr;
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
  PyType_mdarray = queryPyTypeObject("_p_mdarray");

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

#define SWIG_as_voidptr(a) const_cast< void * >(static_cast< const void * >(a)) 
  PyObject *mdarray::m_Add(PyObject *self, PyObject *o) {
    void *oprd2;
    int res = SWIG_ConvertPtr(o, &oprd2, nullptr, 0);
    if (!SWIG_IsOK(res)) {
      PyErr_SetString(PyExc_ValueError, "Wrong operand object in add wrapper");
      return nullptr;
    }
    // mdarray + ndarray
    if (strcmp(o->ob_type->tp_name, "mkldnn.mdarray.mdarray")) {
        PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0
                , NPY_ARRAY_ELEMENTSTRIDES, nullptr);

        if (surrogate == nullptr)
            return nullptr;

        PyObject *res = PyNumber_Add(surrogate, o);
        Py_DECREF(surrogate);
        return res;
    }

    // 2 mdarray add
    std::vector<double> scale;
    std::vector<mkldnn::memory::primitive_desc> inputs_mpd;
    std::vector<mkldnn::memory::primitive::at> inputs_at;
    py_handle *output = new py_handle();
    mdarray *mdarray1 = (mdarray *)this;
    auto mdarray2 = (reinterpret_cast<py_handle*>(oprd2))->get();
    mkldnn::memory::primitive_desc lsrc1_mpd = mdarray1->m_.get_primitive_desc();
    mkldnn::memory::primitive_desc lsrc2_mpd = mdarray2->m_.get_primitive_desc();
    scale.push_back((double)1.0);
    scale.push_back((double)1.0);
    inputs_mpd.push_back(lsrc1_mpd);
    inputs_mpd.push_back(lsrc2_mpd);
    auto sum_pd = mkldnn::sum::primitive_desc(scale, inputs_mpd);
    inputs_at.push_back(mdarray1->memory());
    auto m = mdarray2->memory();
    inputs_at.push_back(mdarray2->memory());
    mkldnn::memory::primitive_desc dst_mpd = sum_pd.dst_primitive_desc();
    output->reset(new mdarray(dst_mpd));
    auto sum_prim = mkldnn::sum(sum_pd, inputs_at, (*output)->memory());

    std::vector<mkldnn::primitive> sum_prims;
    sum_prims.push_back(sum_prim);

    auto stream = mkldnn::stream(mkldnn::stream::kind::eager);
    stream.submit(sum_prims).wait();
    auto resultobj = SWIG_Python_NewPointerObj(nullptr, SWIG_as_voidptr(output), SWIG_TypeQuery("_p_mdarray"), SWIG_POINTER_OWN |  0 );
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
