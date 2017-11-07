#pragma once

#include <stdlib.h>
#include <mutex>
#include <list>
#include <functional>
#include <sstream>
#include <string>
#include <memory>
#include "utils.h"

typedef enum {
    MPOOL_WEIGHT,
    MPOOL_IP,
    MPOOL_CONV,
    MPOOL_DECONV,
    MPOOL_RELU,
    MPOOL_POOLING,
    MPOOL_BATCHNORM,
    MPOOL_LRN,
    MPOOL_CONCAT,
    MPOOL_ANON,
} mem_pool_t;

template <std::size_t ALIGNMENT>
class Memory {
public:
    void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        void *ptr;
        int idx = to_index(size);

        if (!free_hashline_[idx].empty()) {
            block_t *block = nullptr; 
            std::list<block_t *> &list = free_hashline_[idx];
            typename std::list<Memory<ALIGNMENT>::block_t*>::iterator it;
            for(it=list.begin(); it != list.end(); ++it) {
                block_t *block = *it; 
                if(block->header_.size_ == size) {
                    break;
                }
            }
            list.erase(block);
            void *ptr = static_cast<void *>(block);
            return (ptr + ALIGNMENT);
        }
        // No cached memory
        size_t len = size + ALIGNMENT;
        int rc = ::posix_memalign(&ptr, ALIGNMENT, len);
        if (rc != 0) {
            throw std::invalid_argument("Out of memory");
        }
        block_t *block = static_cast<block_t *>(ptr); 
        block->header_.size_ = size;
        return (ptr + ALIGNMENT);
    }

    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        block_t *block = static_cast<block_t *>(ptr - ALIGNMENT);
        int idx = to_index(block->header_.size_);
        free_hashline_[idx].push_bash(block);
    }

    void epoch() {
    }
private:
    int to_index(std::size_t size) {
        std::string str = long_to_string(size);
        std::size_t hash = std::hash<std::string>{}(str);
        int idx = hash % MAX_ENTRY;
        return idx;
    }

    typedef union _header_str {
        struct {
            std::size_t size_;
        };
        char pad_[ALIGNMENT];
    } header_t;

    typedef struct _block_str {
        header_t header_;
        char data_[];
    } block_t;

    static constexpr int MAX_ENTRY = 512;

    int alignment_;
    size_t total_size_;
    std::list<block_t *> free_hashline_[MAX_ENTRY];
    std::mutex mutex_;
};

Memory<64> conv_pool();

void* dnn_malloc(size_t size, mem_pool_t pool=MPOOL_ANON);
void* dnn_free(void*p, mem_pool_t pool=MPOOL_ANON);

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
