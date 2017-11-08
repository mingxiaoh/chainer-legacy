#include "mem.h"

static Memory<DEFAULT_ALIGNMENT> anon_pool;

void* dnn_malloc(size_t size, mem_pool_t mpool)
{
    return anon_pool.malloc(size);
}

void dnn_free(void *p, mem_pool_t mpool)
{
    return anon_pool.free(p);
}
