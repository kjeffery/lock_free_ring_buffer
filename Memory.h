//
// Created by Keith on 12/4/2020.
//

#pragma once

#include <cstddef>

#if defined(_MSC_FULL_VER)
#include <malloc.h>
#include <windows.h>
inline std::size_t get_page_size()
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

inline void* aligned_allocate(std::size_t size, std::size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

inline void aligned_deallocate(void* p)
{
    _aligned_free(p);
}
#else
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <stdlib.h>
#include <unistd.h>
inline std::size_t get_page_size()
{
    return getpagesize();
}

inline void* aligned_allocate(std::size_t size, std::size_t alignment)
{
#if __cplusplus >= 201703L
    return std::aligned_alloc(alignment, size);
#else
    void* mem;
    const auto e = posix_memalign(&mem, alignment, size);
    assert(e != EINVAL);
    if (e == ENOMEM) {
        return nullptr;
    }
    return mem;
#endif
}

inline void aligned_deallocate(void* p)
{
    free(p);
}
#endif

constexpr std::size_t k_cache_line_size = 64u;


