#ifndef _MDARRAY_H_
#define _MDARRAY_H_
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <mkldnn.hpp>
#include <type_traits>

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

// template<class Impl> 
// class py_interf {
// public:
//   py_interf(): pImpl_(nullptr) {}
//   py_interf(const py_interf &orig): pImpl_(orig.pImpl_) {}
//   py_interf(Impl *impl): pImpl_(impl) {}
//   py_interf(std::shared_ptr<Impl> impl): pImpl_(impl) {}
//   Impl *get() {
//     return pImpl_.get();
//   }
// protected:
//   std::shared_ptr<Impl> pImpl_;
// };

using py_handle = std::shared_ptr<implementation::mdarray>;

template <class to>
static bool isa(const py_handle &t) {
  return to::classof(t.get());
}

namespace implementation {

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
  static constexpr int MAX_NDIM = 12; //XXX: For now
  typedef size_t size_type;
  // Generated on demand
  struct _data_desc {
    int ndims;
    char format[4];
    Py_ssize_t itemsize;
    Py_ssize_t strides[MAX_NDIM];
    Py_ssize_t shape[MAX_NDIM];
  };

  mdarray(mkldnn::memory::dims dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine) : 
              size_(std::accumulate(dims.begin(), dims.end(), 1
                    , std::multiplies<mkldnn::memory::dims::value_type>()))
              , data_(new avx::byte [size_ * 4])
              , m_({{dims, dt, format}, engine}, data_.get())
              , desc_(nullptr), view_(nullptr), rtti(raw) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : size_([] (mkldnn::memory::primitive_desc &pd) {
                    auto md = pd.desc().data;
                    return std::accumulate(md.dims, md.dims + md.ndims, 1
                        , std::multiplies<int>());
                  }(pd))
              , data_(new avx::byte [size_ * 4])
              , m_(pd, data_.get())
              , desc_(nullptr), view_(nullptr), rtti(raw) {}
  
  mdarray(Py_buffer *view
      , mkldnn::memory::format format
      , mkldnn::engine &e)
    : size_(view->len/view->itemsize)
          , data_ ([](Py_buffer *view) {
             unsigned long adrs = reinterpret_cast<unsigned long>(view->buf);
             if (adrs % 16 != 0) {
               return std::unique_ptr
                 <avx::byte []>(new avx::byte [view->len]);
             } else
               return std::unique_ptr<avx::byte []>(nullptr);
           } (view))
          , m_({d_from_view(view, format), e}
              , data_ == nullptr? view->buf : data_.get())
          , desc_(nullptr), view_(view), rtti(raw) {
    if (data_ != nullptr) {
      // XXX: OpenMP thing?
      memcpy(data_.get(), view->buf, view->len);
      view_.reset(nullptr);
    }
  }

  inline void *data() { return view_ == nullptr ? data_.get(): view_->buf; }
  inline size_type size() { return size_; }

  inline int ndims() {
    auto md = m_.get_primitive_desc().desc();
    return md.data.ndims;
  }

  inline mkldnn::memory &memory() {
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

private:
  struct WeDontManageIt {
    void operator() (Py_buffer *view) {
      PyBuffer_Release(view);
      delete view;
    }
  };

  // Attributes
  size_type size_;
  std::unique_ptr<avx::byte []> data_;
  mkldnn::memory m_;
  std::unique_ptr<_data_desc> desc_;
  std::unique_ptr<Py_buffer, WeDontManageIt> view_;

protected:
  enum mdarray_ty{
    raw, dual_out
  };
  mdarray_ty rtti;
public:
  static bool classof(const mdarray *p) {
    return p->get_kind() == raw;
  }

  mdarray_ty get_kind() const { return rtti; }
private:
  // Private helpers
  void _collect_buffer_info() {
    if (desc_ == nullptr)
      desc_ = std::unique_ptr<_data_desc>(new _data_desc);

    // XXX: Do we need collect information every time?
    // For safety we do now.

    auto md = m_.get_primitive_desc().desc();
    int ndims = md.data.ndims;

    desc_->ndims = ndims;
    switch(md.data.data_type) {
      case mkldnn::memory::f32:
        strcpy(desc_->format, "f");
        desc_->itemsize = 4;
        break;
      case mkldnn::memory::s32:
        strcpy(desc_->format, "i");
        desc_->itemsize = 4;
        break;
      default:
        break;
    }

    // XXX: figure this out
    for (int i = 0; i < ndims; i ++) {
      desc_->shape[i] = md.data.dims[i];
    }

    Py_ssize_t sd = desc_->itemsize;

    for (int i = ndims -1; i >= 0; --i) {
      desc_->strides[i] = sd;
      sd *= desc_->shape[i];
    }
  }

  mkldnn::memory::desc d_from_view(Py_buffer *view
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
public:

};

int mdarray::getbuffer(PyObject *self, Py_buffer *view, int flags) {
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
    PyErr_SetString(PyExc_ValueError, "carray is not Fortran contiguous");
    goto fail;
  }

  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    goto fail;
  }

  /* Fill in buffer detail */
  _collect_buffer_info();

  view->buf = data();
  view->itemsize = desc_->itemsize;
  view->readonly = 0;
  view->internal = nullptr;
  view->len = size_ * desc_->itemsize;

  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    view->format = desc_->format;
  } else {
    view->format = nullptr;
  }

  if ((flags & PyBUF_ND) == PyBUF_ND) {
    view->ndim = desc_->ndims;
    view->shape = desc_->shape;
  } else {
    view->ndim = 0;
    view->shape = nullptr;
  }

  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
    view->strides = desc_->strides;
  } else
    view->strides = nullptr;

  // We do not have to check PyBUF_INDIRECT because we
  // are C contiguous
  view->suboffsets = nullptr;

  view->obj = self;
  Py_INCREF(self);

  return 0;

fail:
  return -1;
}

PyObject *mdarray::getattro(PyObject *self, PyObject *name) {
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

    // Switch to our message if things gone wrong
    PyTypeObject *tp = Py_TYPE(self);
    PyErr_Format(PyExc_AttributeError
        , "'%.50s' object has no attribute '%U'", tp->tp_name, name);
  }

  return attr;
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

static memory reorder_if_must(mkldnn::memory user
    , mkldnn::memory::primitive_desc expect
    , std::vector<primitive> *dag) {

  if (user.get_primitive_desc() != expect) {
    mkldnn::memory interm(expect);

    dag->push_back(reorder(user, interm));
    return interm;
  }

  return user;
}

template <class p_t
, typename pd_t = typename p_t::primitive_desc>
class f_s_op: public s_op {
private:
  f_s_op(pd_t &op, mdarray *x, mdarray *W, mdarray *b
      , std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag), interms_(2) {

    mkldnn::memory x_interm = reorder_if_must(x->memory()
        , op.src_primitive_desc(), dag_);
    mkldnn::memory W_interm = reorder_if_must(W->memory()
        , op.weights_primitive_desc(), dag_);

    dag_->push_back(p_t(op, x_interm, W_interm
          , b->memory(), this->memory()));

    interms_.push_back(x_interm);
    interms_.push_back(W_interm);
  }

  f_s_op(pd_t &op, mdarray *x, mdarray *W
      , std::vector<primitive> *dag)
    : s_op(op.dst_primitive_desc(), dag) {

    mkldnn::memory x_interm (reorder_if_must(x->memory()
          , op.src_primitive_desc(), dag_));
    mkldnn::memory W_interm (reorder_if_must(W->memory()
          , op.weights_primitive_desc(), dag_));

    dag_->push_back(p_t(op, x_interm, W_interm, this->memory()));

    interms_.push_back(x_interm);
    interms_.push_back(W_interm);
  }

public:
  f_s_op(pd_t &op, py_handle x, py_handle W, py_handle b
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), b.get(), dag){
      deps_ = {x, W, b};
    }

  f_s_op(pd_t &op, py_handle x, py_handle W
      , std::vector<primitive> *dag)
    : f_s_op(op, x.get(), W.get(), dag){
      deps_= {x, W};
    }

private:
  std::vector<mkldnn::primitive> interms_;
  std::vector<py_handle> deps_;
};

template <class p_t, typename pd_t = typename p_t::primitive_desc>
class bd_op: public s_op {
private:
  bd_op(pd_t &op
      , mdarray *gy, mdarray *W, std::vector<primitive> *dag)
    : s_op(op.diff_src_primitive_desc(), dag), interms_(2) {

    mkldnn::memory gy_interm (reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), dag_));

    mkldnn::memory W_interm (reorder_if_must(W->memory()
          , op.weights_primitive_desc(), dag_));

    dag_->push_back(p_t(op, gy_interm, W_interm
          , this->memory()));

    interms_.push_back(gy_interm);
    interms_.push_back(W_interm);
  }

public:
  bd_op(pd_t &op, py_handle gy, py_handle W
      , std::vector<primitive> *dag)
    : bd_op(op, gy.get(), W.get(), dag) {
      deps_={gy, W};
    }

private:
  std::vector<mkldnn::primitive> interms_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bwb_op: public d_op {
public:
  bwb_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : d_op(op.diff_weights_primitive_desc()
        , op.diff_bias_primitive_desc(), dag), interms_(2) {

    mkldnn::memory x_interm (reorder_if_must(x->memory()
          , op.src_primitive_desc(), dag_));

    mkldnn::memory gy_interm (reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), dag_));

    dag_->push_back(p_t(op, x_interm, gy_interm
          , memory(), extra->memory()));

    interms_.push_back(x_interm);
    interms_.push_back(gy_interm);
  }
public:
  bwb_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bwb_op(op, x.get(), gy.get(), dag) { deps_ = {x, gy}; }

private:
  std::vector<mkldnn::primitive> interms_;
  std::vector<py_handle> deps_;
};

template<class p_t, typename pd_t = typename p_t::primitive_desc>
class bw_op: public s_op {
public:
  bw_op(pd_t &op
      , mdarray *x, mdarray *gy, std::vector<primitive> *dag)
    : s_op(op.diff_weights_primitive_desc(), dag), interms_(2) {

    mkldnn::memory x_interm (reorder_if_must(x->memory()
          , op.src_primitive_desc(), dag_));

    mkldnn::memory gy_interm (reorder_if_must(gy->memory()
          , op.diff_dst_primitive_desc(), dag_));

    dag_ ->push_back(p_t(op, x_interm, gy_interm
          , memory()));

    interms_.push_back(x_interm);
    interms_.push_back(gy_interm);
  }
public:
  bw_op(pd_t &op, py_handle x, py_handle gy
      , std::vector<primitive> *dag)
    : bw_op(op, x.get(), gy.get(), dag) { deps_ = {x, gy}; }
private:
  std::vector<mkldnn::primitive> interms_;
  std::vector<py_handle> deps_;
};

}

// Actual interface for python
// DO NOT add field beyond py_handle
//
class mdarray : public py_handle {
public:
  mdarray(mkldnn::memory::dims dims
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

};

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

  static py_handle extra_get(py_handle &in) {
    if (isa<implementation::d_op>(in)){
        return implementation::d_op::extra_get
        (reinterpret_cast<implementation::d_op *>(in.get()));
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

#endif
