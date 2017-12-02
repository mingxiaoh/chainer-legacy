#include <vector>
#include "mem.h"

using namespace std;

#define MALLOC_FREE_IMPL(prefix) \
    static Memory<DEFAULT_ALIGNMENT> prefix##_pool(#prefix); \
    avx::byte* prefix##_malloc(size_t size) { \
        return (avx::byte *)prefix##_pool.malloc(size); \
    } \
    void prefix##_free(avx::byte *p) { \
        return prefix##_pool.free((void *)p); \
    }

MALLOC_FREE_IMPL(anon)
MALLOC_FREE_IMPL(reorder)
MALLOC_FREE_IMPL(relu_fwd)
MALLOC_FREE_IMPL(relu_bwd)
MALLOC_FREE_IMPL(bn_fwd)
MALLOC_FREE_IMPL(bn_bwd)
MALLOC_FREE_IMPL(lrn_fwd)
MALLOC_FREE_IMPL(lrn_bwd)

std::shared_ptr<avx::byte> Allocator::malloc(size_t len, mem_pool_t mpool)
{
    std::shared_ptr<avx::byte> data;
    switch(mpool) {
        case MPOOL_REORDER:
            data = std::shared_ptr<avx::byte>(reorder_malloc(len), reorder_free);
            break;
        case MPOOL_RELU_FWD:
            data = std::shared_ptr<avx::byte>(relu_fwd_malloc(len), relu_fwd_free);
            break;
        case MPOOL_RELU_BWD:
            data = std::shared_ptr<avx::byte>(relu_bwd_malloc(len), relu_bwd_free);
            break;
        case MPOOL_BN_FWD:
            data = std::shared_ptr<avx::byte>(bn_fwd_malloc(len), bn_fwd_free);
            break;
        case MPOOL_BN_BWD:
            data = std::shared_ptr<avx::byte>(bn_bwd_malloc(len), bn_bwd_free);
            break;
        case MPOOL_LRN_FWD:
            data = std::shared_ptr<avx::byte>(lrn_fwd_malloc(len), lrn_fwd_free);
            break;
        case MPOOL_LRN_BWD:
            data = std::shared_ptr<avx::byte>(lrn_bwd_malloc(len), lrn_bwd_free);
            break;
        default:
            data = std::shared_ptr<avx::byte>(anon_malloc(len), anon_free);
            break;
    }

    return data;
}

std::shared_ptr<avx::byte> Allocator::malloc(vector<int> dims, int element_sz, mem_pool_t mpool)
{
    auto len = std::accumulate(dims.begin(), dims.end(), 1
            , std::multiplies<int>()) * element_sz;

    return Allocator::malloc(len, mpool);
}

void* dnn_malloc(size_t size, mem_pool_t mpool)
{
    return anon_pool.malloc(size);
}

void dnn_free(void *p, mem_pool_t mpool)
{
    return anon_pool.free(p);
}
