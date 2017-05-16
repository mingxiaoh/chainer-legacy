#ifndef _MDARRAY_H_
#define _MDARRAY_H_
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <mkldnn.hpp>
#include <type_traits>
#if !defined(SWIG_INLINE)
#include <swigpyrun.h>
#endif

// Just grab it from MKL-DNN
namespace avx {
  inline void* malloc(size_t size, int alignment) {
      void *ptr;
      int rc = ::posix_memalign(&ptr, alignment, size);
      return (rc == 0) ? ptr : 0;
  }
  inline void free(void* p) { ::free(p); }

  struct compatible {
      enum { default_alignment = 64 };
      static void* operator new(size_t sz) {
          return malloc(sz, default_alignment);
      }
      static void* operator new(size_t sz, void* p) { (void)sz; return p; }
      static void* operator new[](size_t sz) {
          return malloc(sz, default_alignment);
      }
      static void operator delete(void* p) { free(p); }
      static void operator delete[](void* p) { free(p); }
  };

  struct byte: public compatible {
    char q;
  };
}

namespace implementation {
  class mdarray;
}

using py_handle = std::shared_ptr<implementation::mdarray>;

template <class to>
static bool isa(const py_handle &t) {
  return to::classof(t.get());
}

namespace implementation {
static PyObject *PyType_reorder_buffer = nullptr;

  // We brought this to global scope to mitigate it consumption
  void g_init() {
    swig_type_info *Py_reorder_buffer = SWIG_TypeQuery("_p_reorder_buffer");
    if (Py_reorder_buffer != nullptr) {
      SwigPyClientData *cd
        = (SwigPyClientData *)Py_reorder_buffer->clientdata;
      PyType_reorder_buffer = reinterpret_cast<PyObject *>(cd->pytype);
    }

    if (PyType_reorder_buffer == nullptr)
      throw mkldnn::error(mkldnn::c_api::mkldnn_invalid_arguments
          , "Failed to find reorder_buffer object");
  }

#define nb_unary_map(method) \
  PyObject * m_ ## method (PyObject *self) {    \
    PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
        , NPY_ARRAY_ELEMENTSTRIDES, nullptr);   \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate); \
    Py_DECREF(surrogate);   \
    return res;   \
  }

#define nb_binary_map(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o) {    \
    PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
        , NPY_ARRAY_ELEMENTSTRIDES, nullptr);   \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o); \
    Py_DECREF(surrogate);   \
    return res;   \
  }

#define nb_ternary_map(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o1, PyObject *o2) {    \
    PyObject *surrogate = PyArray_FromAny(self, nullptr, 0, 0 \
        , NPY_ARRAY_ELEMENTSTRIDES, nullptr);   \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o1, o2); \
    Py_DECREF(surrogate); \
    return res;   \
  }

class mdarray {
public:
  // It is exposed to python
  //
  static constexpr int MAX_NDIM = 12; //XXX: For now

  class reorder_buffer {
  private:
    mkldnn::memory dst_;
    std::shared_ptr<avx::byte> data_;

    int ndims_;
    int size_;
    char format_[4];
    Py_ssize_t itemsize_;
    Py_ssize_t strides_[MAX_NDIM];
    Py_ssize_t shape_[MAX_NDIM];

    void _collect_buffer_info() {
      auto md = dst_.get_primitive_desc().desc();
      int ndims = md.data.ndims;

      ndims_ = ndims;
      switch(md.data.data_type) {
        case mkldnn::memory::f32:
          strcpy(format_, "f");
          itemsize_ = 4;
          break;
        case mkldnn::memory::s32:
          strcpy(format_, "i");
          itemsize_ = 4;
          break;
        default:
          break;
      }

      for (int i = 0; i < ndims; i ++) {
        shape_[i] = md.data.dims[i];
      }

      Py_ssize_t sd = itemsize_;

      for (int i = ndims -1; i >= 0; --i) {
        strides_[i] = sd;
        sd *= shape_[i];
      }
    }

  public:
    reorder_buffer(std::shared_ptr<mdarray> in)
      :reorder_buffer(in.get()) {}

    reorder_buffer(const mdarray *src)
      : dst_([src] () {
          if (src->internal()) {
            auto md_data = src->desc().data;

            mkldnn::memory::dims adims(md_data.dims
                , md_data.dims + md_data.ndims);

            mkldnn::memory::primitive_desc pd ({adims
                , static_cast<mkldnn::memory::data_type>(md_data.data_type)
                , public_format(md_data.ndims)}
                // Added interface for it
                , src->get_engine());

            // XXX: magic number 4 is a hack
            return mkldnn::memory(pd, reinterpret_cast<void *>(4));
          } else {
            return src->memory();
          }} ()), size_(src->size()) {
        if (src->internal()) {
          auto pd = dst_.get_primitive_desc();

          data_ = std::shared_ptr<avx::byte>(new avx::byte [pd.get_size()]
              , [](avx::byte *p) {delete [] p;});

          dst_.set_data_handle(data_.get());

          mkldnn::reorder reorder(src->memory(), dst_);
          mkldnn::stream s(mkldnn::stream::eager);
          s.submit({reorder});

        } else {
          data_ = src->share_data();
        }

        _collect_buffer_info();
      }

    int build_view(Py_buffer *view, int flags) {
      if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
        PyErr_SetString(PyExc_ValueError, "C contiguous buffer only!");
        return -1;
      }

      if (view == nullptr) {
        PyErr_SetString(PyExc_ValueError, "NULL view.");
        return -1;
      }

      view->buf = data_.get();
      view->itemsize = itemsize_;
      view->readonly = 0;
      view->internal = nullptr;
      view->len = size_ * itemsize_;

      if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = format_;
      } else {
        view->format = nullptr;
      }

      if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = ndims_;
        view->shape = shape_;
      } else {
        view->ndim = 0;
        view->shape = nullptr;
      }

      if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->strides = strides_;
      } else {
        view->strides = nullptr;
      }

      view->suboffsets = nullptr;

      return 0;
    }

    static mkldnn::memory::format public_format(int ndims) {
      switch(ndims) {
      case 2:
        return mkldnn::memory::nc;
        break;
      case 4:
        return mkldnn::memory::nchw;
        break;
      }

      return mkldnn::memory::format_undef;
    }
  };

public:
  typedef size_t size_type;
  // Generated on demand
  virtual ~mdarray() {}

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine)
    : mdarray({{std::move(dims), dt, format}, engine}) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : size_([&pd] () {
                    auto md = pd.desc().data;
                    return std::accumulate(md.dims, md.dims + md.ndims, 1
                        , std::multiplies<int>());
                  }())
              // Use primitive desc's reference
              , data_(new avx::byte [pd.get_size()]
                  , [](avx::byte *p) {delete [] p;})
              , m_(pd, data_.get())
              , view_(nullptr), rtti(raw)
              , internal_order_([&pd] () {
                  auto md = pd.desc().data;
                  if (md.format != mkldnn::memory::x
                      && md.format != mkldnn::memory::nc
                      && md.format != mkldnn::memory::nchw
                      && md.format != mkldnn::memory::oi
                      && md.format != mkldnn::memory::oihw) { 
                    std::cout<<"Weired format "<<md.format<<std::endl;
                    return true;
                  }
                  else
                    return false;
                  } ()) {}
  
  mdarray(Py_buffer *view
      , mkldnn::memory::format format
      , mkldnn::engine &e)
    : size_(view->len/view->itemsize)
          , data_ ([view]() {
             unsigned long adrs = reinterpret_cast<unsigned long>(view->buf);
             if (adrs % 16 != 0) {
               return std::shared_ptr<avx::byte>(new avx::byte [view->len]
                   , [] (avx::byte *p) {delete [] p;});
             } else
               return std::shared_ptr<avx::byte>(reinterpret_cast<avx::byte *>(view->buf)
                   , [] (avx::byte *p) {});
           } ())
          , m_({_d_from_view(view, format), e}, data_.get())
          , view_(view), rtti(raw), internal_order_(false) {
    if (data_.get() != view->buf) {
      // XXX: Add OpenMP thing?
      memcpy(data_.get(), view->buf, view->len);
      view_.reset(nullptr);
    }
  }

  // TODO: for view case, shared buffer won't expand life in this case
  // because mdarray will destroy it when out of service.
  //
  int setbuffer(Py_buffer *view) {
//    if (desc_ != nullptr)
//      // TODO: not support by provided buffer to numpy
//      goto fail;
//    else {
      // TODO: Guard this section with asserts
      view_.reset(view);

      unsigned long adrs = reinterpret_cast<unsigned long>(view->buf);

      if (adrs % 16 != 0) {
        data_.reset(new avx::byte [view->len]
            , [] (avx::byte *p) {delete [] p;});
        memcpy(data_.get(), view->buf, view->len);
        view_.reset(nullptr);
      } else
        data_.reset(reinterpret_cast<avx::byte *>(view->buf)
            , [] (avx::byte *p) {});

      m_.set_data_handle(data());
//    }

    return 0;
//  fail:
//    return -1;
  }

  inline void *data() const { return data_.get(); }
  inline size_type size() const { return size_; }
  inline size_type len() const { return m_.get_primitive_desc().get_size(); }
  inline mkldnn::engine get_engine() const {
    return m_.get_primitive_desc().get_engine();
  }

  inline int ndims() const {
    auto md = m_.get_primitive_desc().desc();
    return md.data.ndims;
  }

  inline mkldnn::memory memory() const {
    return m_;
  }
  
  inline mkldnn::memory::desc desc() const {
    return m_.get_primitive_desc().desc();
  }

  // PEP: 3118 Buffer Protocol Producer
  int getbuffer(PyObject *obj, Py_buffer *view, int flags);
  PyObject *getattro(PyObject *self, PyObject *name);

  // Do not support old Buffer Protocol
  Py_ssize_t getsegcount(PyObject *self, Py_ssize_t *lenp) {
    return 0;
  }
  Py_ssize_t getreadbuf(PyObject *self, Py_ssize_t segment, void **ptrptr) {
    return 0;
  }
  Py_ssize_t getwritebuf(PyObject *self, Py_ssize_t segment, void **ptrptr) {
    return 0;
  }
  Py_ssize_t getcharbuf(PyObject *self, Py_ssize_t segment, void **ptrptr) {
    return 0;
  }

  nb_binary_map(Add);
  nb_binary_map(Subtract);
  nb_binary_map(Multiply);
  nb_binary_map(Remainder);
  nb_binary_map(Divmod);
  nb_unary_map(Negative);
  nb_unary_map(Positive);
  nb_unary_map(Absolute);
  nb_unary_map(Invert);
  nb_binary_map(Lshift);
  nb_binary_map(Rshift);
  nb_binary_map(And);
  nb_binary_map(Xor);
  nb_binary_map(Or);
  nb_binary_map(InPlaceAdd);
  nb_binary_map(InPlaceSubtract);
  nb_binary_map(InPlaceMultiply);
  nb_binary_map(InPlaceRemainder);
  nb_ternary_map(InPlacePower);
  nb_binary_map(InPlaceLshift);
  nb_binary_map(InPlaceRshift);
  nb_binary_map(InPlaceAnd);
  nb_binary_map(InPlaceXor);
  nb_binary_map(InPlaceOr);
  nb_binary_map(FloorDivide);
  nb_binary_map(TrueDivide);
  nb_binary_map(InPlaceFloorDivide);
  nb_binary_map(InPlaceTrueDivide);
  nb_binary_map(MatrixMultiply);
  nb_binary_map(InPlaceMatrixMultiply);

  Py_ssize_t mp_length(PyObject *self);
  PyObject *mp_subscript(PyObject *self, PyObject *op);
  int mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op);

private:
  struct WeDontManageIt {
    void operator() (Py_buffer *view) {
      PyBuffer_Release(view);
      delete view;
    }
  };

  // Attributes
  size_type size_;
  std::shared_ptr<avx::byte> data_;
  mkldnn::memory m_;
  std::unique_ptr<Py_buffer, WeDontManageIt> view_;

protected:
  enum mdarray_ty{
    raw, dual_out
  };
  mdarray_ty rtti;
  bool internal_order_;

public:
  bool internal() const { return internal_order_; }
  std::shared_ptr<avx::byte> share_data() const {
    return data_;
  }

  static bool classof(const mdarray *p) {
    return p->get_kind() == raw;
  }

  mdarray_ty get_kind() const { return rtti; }

  static mkldnn::memory reorder_if_must(mkldnn::memory user
      , mkldnn::memory::primitive_desc expect
      , std::vector<mkldnn::primitive> *dag) {

    if (user.get_primitive_desc() != expect) {
      mkldnn::memory interm(expect);

      dag->push_back(mkldnn::reorder(user, interm));
      return interm;
    }

    return user;
  }

protected:
  // Private helpers
private:
  static mkldnn::memory::desc _d_from_view(Py_buffer *view
      , mkldnn::memory::format order) {
    mkldnn::memory::dims dims (view->ndim);

    for( int i=0; i < view->ndim; i++)
      dims[i] = view->shape[i];

    std::string format(view->format);
    mkldnn::memory::data_type dt; 

    if (view->itemsize == 4) {
      if (std::string::npos != format.find_last_of('f')) {
        dt = mkldnn::memory::f32;
      } else if (std::string::npos != format.find_last_of('i')) {
        dt = mkldnn::memory::s32;
      } else
        throw mkldnn::error(mkldnn::c_api::mkldnn_invalid_arguments
            , std::string("MKLDNN does not support data type: ")
            + format);
    } else
      throw mkldnn::error(mkldnn::c_api::mkldnn_invalid_arguments
          , "MKLDNN does not support itemsize other than 4");

    return mkldnn::memory::desc(dims, dt, order);
  }
};

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


// XXX: solve dual outputs problem
// Type system should be rework

class s_op: public mdarray {
public:
  s_op(mkldnn::memory::primitive_desc dst
      , std::vector<mkldnn::primitive> *dag)
    : mdarray(dst), dag_(dag) {
  }

protected:
  std::vector<mkldnn::primitive> *dag_;
};

class d_op : public mdarray {
public:
  // XXX: Tricky part, how extra managed
  d_op(mkldnn::memory::primitive_desc major
      , mkldnn::memory::primitive_desc extra
      , std::vector<mkldnn::primitive> *dag):
    mdarray(major), extra(new s_op(extra, dag)), dag_(dag) {
    rtti = dual_out;
  }

  static py_handle extra_get(const d_op *that) {
    return that->extra;
  }

  static bool classof(const mdarray *p) {
    return p->get_kind() == dual_out;
  }
protected:
  // This seems unique, but it will share in python
  // Ugly. XXX
  std::shared_ptr<mdarray> extra;
  std::vector<mkldnn::primitive> *dag_;
};

using namespace mkldnn;

template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public s_op {
private:
  f_s_op(pd_t &op, mdarray *x, mdarray *W
      , std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag)
      , x_reordered_(reorder_if_must(x->memory()
            , op.src_primitive_desc(), dag_))
      , W_reordered_(reorder_if_must(W->memory()
            , op.weights_primitive_desc(), dag_)) {
  }

public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), dag) {
      deps_ = {x, W, b};
      dag_->push_back(p_t(op, x_reordered_, W_reordered_
           , b->memory(), this->memory()));
    }

  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), dag){
      deps_= {x, W};
      dag_->push_back(p_t(op, x_reordered_, W_reordered_
            , this->memory()));
    }

private:
  mkldnn::memory x_reordered_, W_reordered_;
  std::vector<py_handle> deps_;
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public s_op {
private:
  bd_op(pd_t &op
      , mdarray *gy, mdarray *W, std::vector<primitive> *dag)
    : s_op(op.diff_src_primitive_desc(), dag)
      , gy_reordered_(reorder_if_must(gy->memory()
            , op.diff_dst_primitive_desc(), dag_))
      , W_reordered_(reorder_if_must(W->memory()
            , op.weights_primitive_desc(), dag_)) {}

public:
  bd_op(pd_t &op, py_handle gy, py_handle W
      , std::vector<primitive> *dag)
    : bd_op(op, gy.get(), W.get(), dag) {
      deps_= {gy, W};
      dag_->push_back(p_t(op, gy_reordered_, W_reordered_
            , this->memory()));
    }

private:
  mkldnn::memory gy_reordered_, W_reordered_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public d_op {
public:
  bwb_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : d_op(op.diff_weights_primitive_desc()
        , op.diff_bias_primitive_desc(), dag)
      , x_reordered_(reorder_if_must(x->memory()
            , op.src_primitive_desc(), dag_))
      , gy_reordered_(reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), dag_)) {}

public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bwb_op(op, x.get(), gy.get(), dag) {
      deps_ = {x, gy};
      dag_->push_back(p_t(op, x_reordered_, gy_reordered_
          , memory(), extra->memory()));
    }

private:
  mkldnn::memory x_reordered_, gy_reordered_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public s_op {
public:
  bw_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : s_op(op.diff_weights_primitive_desc(), dag)
      , x_reordered_(reorder_if_must(x->memory()
          , op.src_primitive_desc(), dag_))
      , gy_reordered_(reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), dag_)) {}
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bw_op(op, x.get(), gy.get(), dag) {
      deps_ = {x, gy};
      dag_ ->push_back(p_t(op, x_reordered_, gy_reordered_
          , memory()));
    }
private:
  mkldnn::memory x_reordered_, gy_reordered_;
  std::vector<py_handle> deps_;
};

}

// Actual interface for python
// DO NOT add any field
//
class mdarray : public py_handle {
public:
  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine)
    : py_handle(new implementation::mdarray(dims
          , dt, format, engine)) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : py_handle(new implementation::mdarray(pd)) {}

  mdarray(Py_buffer *view
      , mkldnn::memory::format format
      , mkldnn::engine &e)
    : py_handle(new implementation::mdarray(view, format, e)) {}

  static PyObject *mdarray_shape_get(mdarray *arg) {
    implementation::mdarray *self = arg->get();
    int ndim = self->ndims();
    PyObject *intTuple = PyTuple_New(ndim);
    auto data = self->desc().data;

    if (!intTuple)
      goto fail;

    for (int i = 0; i<ndim; i++) {
      PyObject *o = PyLong_FromLong(data.dims[i]);
  
      if (!o) {
        Py_DECREF(intTuple);
        intTuple = NULL;
        goto fail;
      }

      PyTuple_SET_ITEM(intTuple, i, o);
    }

  fail:
    return intTuple;
  }

  static PyObject *mdarray_dtype_get(mdarray *self) {
    implementation::mdarray *m = self->get();
    PyArray_Descr *pd;
    // Translate our data_type to numpy one
    switch (m->desc().data.data_type) {
      case mkldnn::memory::f32:
        pd = PyArray_DescrFromType(NPY_FLOAT);
        break;
      case mkldnn::memory::s32:
        pd= PyArray_DescrFromType(NPY_INT);
        break;
      default:
        PyErr_SetString(PyExc_ValueError, "Bad mdarray data_type");
        return nullptr;
    }

    return reinterpret_cast<PyObject *>(pd);
  }

  static long mdarray_size_get(mdarray *self) {
    return self->get()->size();
  }

  static long mdarray_ndim_get(mdarray *self) {
    return self->get()->desc().data.ndims;
  }

  static mkldnn::memory *mdarray_memory_get(mdarray *self) {
    return new mkldnn::memory((*self)->memory());
  }
  
};

using namespace mkldnn;

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class f_s_op : public py_handle {
public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
      , std::vector<primitive> *dag)
    : py_handle(new implementation::f_s_op<p_t, pd_t>
       (op, x, W, b, dag)){}

  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<primitive> *dag)
    : py_handle(new implementation::f_s_op<p_t, pd_t>
       (op, x, W, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op : public py_handle {
public:
  bd_op(pd_t &op, py_handle gy, py_handle W
      , std::vector<primitive> *dag)
    : py_handle (new implementation::bd_op<p_t, pd_t>
        (op, gy, W, dag)) {}
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public py_handle {
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : py_handle (new implementation::bwb_op<p_t, pd_t>
        (op, x, gy, dag)) {}

  static py_handle *extra_get(const py_handle *in) {
    if (isa<implementation::d_op>(*in)){
        return new py_handle(implementation::d_op::extra_get
        (static_cast<implementation::d_op *>(in->get())));
    }

    // Raise exception?
    return nullptr;
  }
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public py_handle {
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : py_handle (new implementation::bw_op<p_t, pd_t>
        (op, x, gy, dag)) {}
};

using reorder_buffer = implementation::mdarray::reorder_buffer;

#endif
