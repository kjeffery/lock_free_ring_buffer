///@author Keith Jeffery

#pragma once

#include <atomic>
#include <thread>

#if defined(_MSC_FULL_VER)
#include <windows.h>
#elif defined (__ICC)
#include <immintrin.h>
#endif

#if !defined(_MSC_FULL_VER)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

inline void do_pause()
{
#if defined(_MSC_FULL_VER)
    YieldProcessor();
#elif defined (__ICC)
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
}

namespace wait_impl
{
template <typename Pred>
inline void spin_lock(Pred pred) noexcept
{
    constexpr int k_pause_iter = 12;
    constexpr int k_yield_iter =  4;

    for (int i = 0; i < k_pause_iter; ++i) {
        if (pred()) {
            return;
        }
        do_pause();
    }

    for (int i = 0; i < k_yield_iter; ++i) {
        if (pred()) {
            return;
        }
        std::this_thread::yield();
    }

    while (true) {
        if (pred()) {
            return;
        }
    }
}
} // namespace wait_impl

template <typename atomic_t>
inline void wait(atomic_t& a, typename atomic_t::value_type old, std::memory_order order = std::memory_order_seq_cst) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.wait(old, order);
#elif defined(linux)
    const auto e = syscall(SYS_futex, std::addressof(a), FUTEX_WAIT_PRIVATE, static_cast<uint32_t>(old), nullptr, nullptr, 0);
    if (!e || errno == EAGAIN) {
        return;
    }
#else
    wait_impl::spin_lock([&a, old, order]() { return a.load(order) != old; });
#endif
}

template <typename atomic_t>
inline void notify_one(atomic_t& a) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.notify_one();
#elif defined(linux)
    syscall(SYS_futex, std::addressof(a), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
#endif
}

template <typename atomic_t>
inline void notify_all(atomic_t& a) noexcept
{
#if defined(__cpp_lib_atomic_wait)
    a.notify_all();
#elif defined(linux)
    syscall(SYS_futex, std::addressof(a), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#endif
}
