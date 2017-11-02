#pragma once

#include <stdlib.h>

class Memory {
public:
    void* alloc(size_t size, int alignment);
    void free(void* p);
};

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
