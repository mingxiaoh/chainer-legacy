%module (package="mkldnn") support
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

namespace mkldnn {

namespace c_api {
  %import c_api.i
}

template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}

protected:
    handle(T t = 0, bool weak = false): _data(0);
    bool operator==(const T other) const;
    bool operator!=(const T other) const;

public:
    handle(const handle &other);
    handle &operator=(const handle &other);
    void reset(T t, bool weak = false);
    T get() const;
    bool operator==(const handle &other) const;
    bool operator!=(const handle &other) const;
};

%template (engine_handle) handle< c_api::mkldnn_engine_t >;

struct engine: public handle<c_api::mkldnn_engine_t> {

    enum kind {
        any = c_api::mkldnn_any_engine,
        cpu = c_api::mkldnn_cpu,
    };

    static size_t get_count(kind akind) {
        return c_api::mkldnn_engine_get_count(convert_to_c(akind));
    }

    engine(kind akind, size_t index);

    // XXX: Solve it! explicit engine(const c_api::mkldnn_engine_t& aengine);
};

%rename (at) primitive::at;
%template (primitive_handle) handle<c_api::mkldnn_primitive_t>;

class primitive: public handle<c_api::mkldnn_primitive_t> {
    using handle::handle;
public:
    struct at {
        c_api::mkldnn_primitive_at_t data;

        at(const primitive &aprimitive, size_t at = 0);
        inline operator primitive() const;
    };

    inline c_api::const_mkldnn_primitive_desc_t get_primitive_desc() const;
};

struct error: public std::exception {
    c_api::mkldnn_status_t status;
    std::string message;
    primitive error_primitive;

    error(c_api::mkldnn_status_t astatus, std::string amessage,
            c_api::mkldnn_primitive_t aerror_primitive = 0);

    static void wrap_c_api(c_api::mkldnn_status_t status,
            std::string message,
            c_api::mkldnn_primitive_t *error_primitive = 0);
};

}
