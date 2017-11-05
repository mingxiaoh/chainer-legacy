#ifndef _MDARRAY_H_
#define _MDARRAY_H_
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <forward_list>
#include <stdexcept>
#include <mkldnn.hpp>
#include <type_traits>
#include <swigpyrun.h>
#include "mem.h"
#include "tensor.h"

// FIXME
// use global engine to init mdarray
using namespace mkldnn;
extern engine cpu_engine;

namespace implementation {
  class mdarray;
}

using py_handle = std::shared_ptr<implementation::mdarray>;

namespace implementation {

#if PY_VERSION_HEX >= 0x03000000
  int g_init();
#else
  void g_init();
#endif

#define NPY_ARRAY_SURROGATE_ENTRY(mdarray) \
  PyObject *surrogate = PyArray_FromAny(mdarray, nullptr, 0, 0 \
      , NPY_ARRAY_ELEMENTSTRIDES, nullptr)   \

#define NPY_ARRAY_SURROGATE_EXIT()

#define nb_unary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self) { \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  } \

#define nb_unary_map(method) \
  nb_unary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self) {    \
    return m_ ## method ## _map_impl(self); \
  } \

#define nb_binary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  }

#define nb_binary_map_impl_with_target_func(method, tfunc) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## tfunc(surrogate, o); \
    Py_DECREF(surrogate);   \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  }

#define nb_binary_map(method) \
  nb_binary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o) {    \
    return m_ ## method ## _map_impl(self, o); \
  } \

#define nb_ternary_map_impl(method) \
  PyObject * m_ ## method ## _map_impl(PyObject *self, PyObject *o1, PyObject *o2) {    \
    NPY_ARRAY_SURROGATE_ENTRY(self); \
                                \
    if (surrogate == nullptr)   \
      return nullptr;           \
                                \
    PyObject *res = PyNumber_ ## method(surrogate, o1, o2); \
    Py_DECREF(surrogate); \
    NPY_ARRAY_SURROGATE_EXIT(); \
    return res;   \
  }

#define nb_ternary_map(method) \
  nb_ternary_map_impl(method) \
  PyObject * m_ ## method (PyObject *self, PyObject *o1, PyObject *o2) {    \
    return m_ ## method ## _map_impl(self, o1, o2); \
  } \

class mdarray : public Tensor {
public:
  // It is exposed to python
  //
  static constexpr int MAX_NDIM = 12; //XXX: For now

  class reorderer {
  protected:
    bool non_trivial_;
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
      switch(static_cast<mkldnn::memory::data_type>(md.data.data_type)) {
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

    inline avx::byte *data() const { return data_.get(); }

  public:
    reorderer(const py_handle in)
      :reorderer(in.get()) {}

    reorderer(const mdarray *src)
      : non_trivial_(src->incompatible()), dst_([src] () {
          if (src->incompatible()) {
            auto md_data = src->desc().data;

            mkldnn::memory::dims adims(md_data.dims
                , md_data.dims + md_data.ndims);

            mkldnn::memory::primitive_desc pd ({adims
                , static_cast<mkldnn::memory::data_type>(md_data.data_type)
                , static_cast<mkldnn::memory::format>
                  (::public_format(md_data.format))}
                , src->get_engine());

            // XXX: magic number 4 is a hack
            return mkldnn::memory(pd, reinterpret_cast<void *>(4));
          } else {
            return (src->memory());
          }} ()), size_(src->size()) {
        if (src->incompatible()) {
          auto pd = dst_.get_primitive_desc();

          data_ = std::shared_ptr<avx::byte>(new avx::byte [pd.get_size()]
              , [](avx::byte *p) {delete [] p;});

          dst_.set_data_handle(data_.get());

        } else {
          data_ = src->share_data();
        }

        _collect_buffer_info();
      }

    mkldnn::reorder fire(const mdarray *src) {
      mkldnn::reorder reorder((src->memory()), dst_);
      mkldnn::stream s(mkldnn::stream::eager);

      s.submit({reorder}).wait();
      return reorder;
    }

    mkldnn::reorder sync(const mdarray *src) {
      mkldnn::reorder reorder(dst_, (src->memory()));
      mkldnn::stream s(mkldnn::stream::eager);

      s.submit({reorder}).wait();
      return reorder;
    }

    inline bool non_trivial() const {
      return non_trivial_;
    }

    // PEP 3118 interface
    int build_view(Py_buffer *view, int flags) {
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

    // Array protocol
    PyArrayInterface *build_array_struct(void) {
      auto arrstr = new PyArrayInterface();

      arrstr->two = 2;
      arrstr->nd = ndims_;
      arrstr->typekind = *((char *)format_);
      arrstr->itemsize = itemsize_;
      arrstr->flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_NOTSWAPPED |
                    NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
      arrstr->flags &= ~(NPY_ARRAY_UPDATEIFCOPY | NPY_ARRAY_OWNDATA);
      arrstr->shape = shape_;
      arrstr->strides = strides_;
      arrstr->data = data_.get();
      arrstr->descr = nullptr;

      return arrstr;
    }
  };

public:
  typedef size_t size_type;
  // Generated on demand
  //FIXME 
  //yli135: add default constructor so that we can pass vector<mdarray> form native
  mdarray();
  virtual ~mdarray() = default;

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , const mkldnn::engine &engine)
    : Tensor(dims, dt, format, engine) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : Tensor(pd) {}

  mdarray(Py_buffer *view) {
    data_type_t dt;
    std::string format(view->format);
    if (view->itemsize == 4) {
      if (std::string::npos != format.find_last_of('f')) {
        dt = FLOAT32;
      } else if (std::string::npos != format.find_last_of('i')) {
        dt = SINT32;
      } else {
        throw mkldnn::error(mkldnn_invalid_arguments
            , std::string("MKLDNN does not support data type: ")
            + format);
      }
    } else {
      throw mkldnn::error(mkldnn_invalid_arguments
          , "MKLDNN does not support itemsize other than 4");
    }
    ndims_ = view->ndim;
    dims_.assign(view->shape, view->shape + view->ndim);
    size_ = view->len / view->itemsize;
    type_ = dt;
    data_ = std::shared_ptr<avx::byte>(new avx::byte [view->len]
                    , [] (avx::byte *p) {delete [] p;});
    memcpy(data_.get(), view->buf, view->len);
    mm_fmt_ = ndims2format(ndims_);
    memory::data_type type = to_mkldnn_type();
    mem_.reset(new mkldnn::memory(
                { { { dims_ }, type, static_cast<memory::format>(mm_fmt_) }
                , cpu_engine }, data_.get()));
  }
  
  inline void unpickled_data(void *pdata) {
    data_.reset(reinterpret_cast<avx::byte *>(pdata));
    //m_.set_data_handle(pdata);
    return;
  }

  inline mkldnn::memory memory() const {
    return *(to_mkldnn_memory());
  }

  PyObject *__getstate__(void) const;

  void __setstate__(PyObject *state);

  PyObject *py_mdarray_from(PyObject *o) const;

  /// d = a * x + b * y, using x's format
  template<class T>
  static void axpby(mdarray *dst, T a, mdarray *x, T b, mdarray *y);

  /// Interface to directly contact python
  template<class T>
  PyObject *axpby(T a, T b, PyObject *o);

  template<class T>
  PyObject *inplace_axpby(T a, PyObject *self, T b, PyObject *o);

  PyObject *flat(void);

  PyObject *m_mult_div(PyObject *self, PyObject *o, int mult_or_div, bool inplace);

  // PEP: 3118 Buffer Protocol Producer
  virtual int getbuffer(PyObject *obj, Py_buffer *view, int flags);

  PyObject *getattro(PyObject *self, PyObject *name);

  PyObject *m_Add(PyObject *self, PyObject *o);
  nb_binary_map_impl(Add);
  PyObject *m_InPlaceAdd(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceAdd);
  PyObject *m_Subtract(PyObject *self, PyObject *o);
  nb_binary_map_impl(Subtract);
  PyObject *m_InPlaceSubtract(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceSubtract);
  PyObject *m_Multiply(PyObject *self, PyObject *o);
  nb_binary_map_impl(Multiply);
  PyObject *m_InPlaceMultiply(PyObject *self, PyObject *o);
  nb_binary_map_impl(InPlaceMultiply);
  // SWIG: nb_true_divide (no slot) <= nb_divide
  PyObject *m_Divide(PyObject *self, PyObject *o);
#if PY_VERSION_HEX < 0x03000000
  nb_binary_map_impl(Divide);
#else
  nb_binary_map_impl_with_target_func(Divide, TrueDivide);
#endif
  PyObject *m_InPlaceDivide(PyObject *self, PyObject *o);
#if PY_VERSION_HEX < 0x03000000
  nb_binary_map_impl(InPlaceDivide);
#else
  nb_binary_map_impl_with_target_func(InPlaceDivide, InPlaceTrueDivide);
#endif

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
  nb_binary_map(InPlaceRemainder);
  nb_ternary_map(InPlacePower);
  nb_binary_map(InPlaceLshift);
  nb_binary_map(InPlaceRshift);
  nb_binary_map(InPlaceAnd);
  nb_binary_map(InPlaceXor);
  nb_binary_map(InPlaceOr);
  nb_binary_map(FloorDivide);
  nb_binary_map(InPlaceFloorDivide);
#if (PY_VERSION_HEX >= 0x03000000)
  nb_binary_map(MatrixMultiply);
  nb_binary_map(InPlaceMatrixMultiply);
#endif

  Py_ssize_t mp_length(PyObject *self);
  PyObject *mp_subscript(PyObject *self, PyObject *op);
  int mp_ass_subscript(PyObject *self, PyObject *ind, PyObject *op);

private:
  struct WeDontManageIt {
    void operator() (const Py_buffer *view) {
      PyBuffer_Release(const_cast<Py_buffer *>(view));
      delete view;
    }
  };

  std::unique_ptr<const Py_buffer, WeDontManageIt> view_;

protected:
  reorderer *sync_reorder_;

public:
  //inline bool incompatible() const { return internal_order_; }
  std::shared_ptr<avx::byte> share_data() const {
    return data_;
  }

  static mkldnn::memory reorder_if_must(mkldnn::memory user
      , mkldnn::memory::primitive_desc expect
      , std::unique_ptr<mkldnn::memory> &mreorder
      , std::vector<mkldnn::primitive> *dag) {

    if (user.get_primitive_desc() != expect) {
      mkldnn::memory interm(expect);
#if 0
      auto user_mpd = user.get_primitive_desc();
      mkldnn::memory::format user_fmt = static_cast<mkldnn::memory::format>(
          user_mpd.desc().data.format);
      mkldnn::memory::format mkl_fmt = static_cast<mkldnn::memory::format>(
          expect.desc().data.format);
      mkldnn::memory::data_type dtype = static_cast<mkldnn::memory::data_type>(
          expect.desc().data.data_type);

      if ((user_fmt == mkldnn::memory::format::nChw16c &&
           mkl_fmt == mkldnn::memory::format::nChw8c) ||
          (mkl_fmt == mkldnn::memory::format::nChw16c &&
           user_fmt == mkldnn::memory::format::nChw8c)) {
          auto m = expect.desc().data;
          int n = m.dims[0], c = m.dims[1], h = m.dims[2], w = m.dims[3];
          mkldnn::memory::dims tz = {n, c, h, w};
          mreorder.reset(new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()}));
          //auto mreorder = new mkldnn::memory({{{ tz }, dtype, mkldnn::memory::format::nchw }, expect.get_engine()});
          auto rep1 = mkldnn::reorder(user, *mreorder);
          auto rep2 = mkldnn::reorder(*mreorder, interm);
          dag->push_back(rep1);
          dag->push_back(rep2);
          //static int spl_nr = 0;
          //printf("\n   %d *Reorder(split) iutput from:%d, to:%d\n", spl_nr++, user_fmt, mkl_fmt);
      } else {
          dag->push_back(mkldnn::reorder(user, interm));
      }
#else
      dag->push_back(mkldnn::reorder(user, interm));
#endif
      return interm;
    }

    return user;
  }

private:
  static mkldnn::memory::desc _d_from_view(const Py_buffer *view
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
        throw mkldnn::error(mkldnn_invalid_arguments
            , std::string("MKLDNN does not support data type: ")
            + format);
    } else
      throw mkldnn::error(mkldnn_invalid_arguments
          , "MKLDNN does not support itemsize other than 4");

    return mkldnn::memory::desc(dims, dt, order);
  }
};

}

//
// Actual interface for python
// DO NOT add field
//
class mdarray : public py_handle {
public:
  //FIXME 
  //yli135: add default constructor so that we can pass vector<mdarray> form native
  mdarray() {};

  mdarray(mkldnn::memory::dims &dims
      , mkldnn::memory::data_type dt
      , mkldnn::memory::format format
      , mkldnn::engine &engine)
    : py_handle(std::make_shared<implementation::mdarray>
        (dims, dt, format, engine)) {}

  mdarray(mkldnn::memory::primitive_desc pd)
    : py_handle(std::make_shared<implementation::mdarray>(pd)) {}

  mdarray(Py_buffer *view)
    : py_handle(std::make_shared<implementation::mdarray>(view)) {}

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
    switch (static_cast<mkldnn::memory::data_type>(m->desc().data.data_type)) {
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

  static bool mdarray_is_mdarray_get(mdarray *self) {
    return true;
  }
};

using reorder_buffer = implementation::mdarray::reorderer;

#endif // _MDARRAY_H_
