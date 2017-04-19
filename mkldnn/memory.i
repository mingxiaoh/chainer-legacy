%module (package="mkldnn") memory
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  using mkldnn::handle_traits;
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i

namespace mkldnn {

namespace c_api {
  %import c_api.i
}

%template (memory_handle) handle<c_api::mkldnn_primitive_desc_t>;

struct memory: public primitive {
private:
    std::shared_ptr<char> _handle;

public:
    typedef std::vector<int> dims; /*manual scrip*/

    template <typename T> static void validate_dims(std::vector<T> v);

    enum data_type {
        data_undef = c_api::mkldnn_data_type_undef,
        f32 = c_api::mkldnn_f32,
        s32 = c_api::mkldnn_s32,
    };

    enum format {
        format_undef = c_api::mkldnn_format_undef,
        any = c_api::mkldnn_any,
        blocked = c_api::mkldnn_blocked,
        x = c_api::mkldnn_x,
        nc = c_api::mkldnn_nc,
        nchw = c_api::mkldnn_nchw,
        nhwc = c_api::mkldnn_nhwc,
        chwn = c_api::mkldnn_chwn,
        nChw8c = c_api::mkldnn_nChw8c,
        nChw16c = c_api::mkldnn_nChw16c,
        oi = c_api::mkldnn_oi,
        io = c_api::mkldnn_io,
        oihw = c_api::mkldnn_oihw,
        ihwo = c_api::mkldnn_ihwo,
        oIhw8i = c_api::mkldnn_oIhw8i,
        oIhw16i = c_api::mkldnn_oIhw16i,
        OIhw8i8o = c_api::mkldnn_OIhw8i8o,
        OIhw16i16o = c_api::mkldnn_OIhw16i16o,
        OIhw8o8i = c_api::mkldnn_OIhw8o8i,
        OIhw16o16i = c_api::mkldnn_OIhw16o16i,
        Ohwi8o = c_api::mkldnn_Ohwi8o,
        Ohwi16o = c_api::mkldnn_Ohwi16o,
        goihw = c_api::mkldnn_goihw,
        gOIhw8i8o = c_api::mkldnn_gOIhw8i8o,
        gOIhw16i16o = c_api::mkldnn_gOIhw16i16o,
        gOIhw8o8i = c_api::mkldnn_gOIhw8o8i,
        gOIhw16o16i = c_api::mkldnn_gOIhw16o16i,
    };

    struct desc {
        friend struct memory;
        c_api::mkldnn_memory_desc_t data;

        desc(dims adims, data_type adata_type,
                format aformat);
        desc(const c_api::mkldnn_memory_desc_t &adata);
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        friend struct memory;
        primitive_desc() {}

        primitive_desc(const desc &adesc, const engine &aengine);

        memory::desc desc();

        size_t get_size() const;

        bool operator==(const primitive_desc &other) const;

        bool operator!=(const primitive_desc &other) const;
    };

    memory(const primitive &aprimitive);
    memory(const primitive_desc &adesc);
    memory(const primitive_desc &adesc, void *ahandle);

    primitive_desc get_primitive_desc() const;
    inline void *get_data_handle() const;
    inline void set_data_handle(void *handle) const;

    // XXX: Trivial, can delete them?
    static c_api::mkldnn_data_type_t convert_to_c(data_type adata_type);
    static c_api::mkldnn_memory_format_t convert_to_c(format aformat);

};

}

%template (dims) std::vector<int>;
