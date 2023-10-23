#pragma once

#include <atomic>
#include <cassert>
#include <cstdio>
#include <type_traits>

//static_assert(__ATOMIC_RELAXED == static_cast<int>(std::memory_order_relaxed));
//static_assert(__ATOMIC_CONSUME == static_cast<int>(std::memory_order_consume));
//static_assert(__ATOMIC_ACQUIRE == static_cast<int>(std::memory_order_acquire));
//static_assert(__ATOMIC_RELEASE == static_cast<int>(std::memory_order_release));
//static_assert(__ATOMIC_ACQ_REL == static_cast<int>(std::memory_order_acq_rel));
//static_assert(__ATOMIC_SEQ_CST == static_cast<int>(std::memory_order_seq_cst));

#if defined(__CUDA_ARCH__)
    //#define FENCE_ACQUIRE __threadfence_system(); __syncthreads()
    //#define FENCE_RELEASE __threadfence_system(); __syncthreads()
    #define FENCE_ACQUIRE __threadfence_system()
    #define FENCE_RELEASE __threadfence_system()
#else
    #define FENCE_ACQUIRE std::atomic_thread_fence(std::memory_order_acquire);
    #define FENCE_RELEASE std::atomic_thread_fence(std::memory_order_release);
#endif

#if !defined(__CUDA_ARCH__)
constexpr
#endif
__host__ __device__ std::memory_order compare_exchange_duo(std::memory_order in) noexcept
{
    switch (in) {
        case std::memory_order_relaxed: return std::memory_order_relaxed;
        case std::memory_order_consume: return std::memory_order_consume;
        case std::memory_order_acquire: return std::memory_order_acquire;
        case std::memory_order_release: return std::memory_order_relaxed;
        case std::memory_order_acq_rel: return std::memory_order_acquire;
        case std::memory_order_seq_cst: return std::memory_order_seq_cst;
    }
    return in;
}

template <typename T>
struct atomic_int_base
{
    using value_type      = T;
    using difference_type = value_type;

    __host__ __device__ constexpr atomic_int_base() noexcept
    : m_value{0}
    {
        store(0);
    }

    __host__ __device__ constexpr atomic_int_base(T desired) noexcept
    : m_value(desired)
    {
        store(desired);
    }

    __host__ __device__ atomic_int_base(const atomic_int_base&)                     = delete;
    __host__ __device__ atomic_int_base& operator=(const atomic_int_base&)          = delete;
    __host__ __device__ atomic_int_base& operator=(const atomic_int_base&) volatile = delete;

    __host__ __device__ T operator=(T desired) noexcept
    {
        store(desired);
        return desired;
    }

    __host__ __device__ T operator=(T desired) volatile noexcept
    {
        store(desired);
        return desired;
    }

    __host__ __device__ void store(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        atomicExch_system(&m_value, desired);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
#else
        //__atomic_store_n(&m_value, desired, static_cast<int>(order));
#endif
    }

    __host__ __device__ void store(T desired, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        atomicExch_system((T*)&m_value, desired);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
#else
        //__atomic_store_n(&m_value, desired, static_cast<int>(order));
#endif
    }

    __host__ __device__ T load(std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAdd_system(const_cast<T*>(&m_value), T(0));
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_load_n(&m_value, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T load(std::memory_order order = std::memory_order_seq_cst) const volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAdd_system(const_cast<T*>(&m_value), T(0));
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_load_n(&m_value, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ operator T() const noexcept
    {
        return load();
    }

    __host__ __device__ operator T() const volatile noexcept
    {
        return load();
    }

    __host__ __device__ T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicExch_system(&m_value, desired);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_exchange_n(&m_value, desired, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T exchange(T desired, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicExch_system((T*)&m_value, desired);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_exchange_n(&m_value, desired, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ bool compare_exchange_weak(T& expected, T desired, std::memory_order success, std::memory_order failure) noexcept
    {
        assert(failure != std::memory_order_release);
        assert(failure != std::memory_order_acq_rel);

#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicCAS_system(&m_value, expected, desired);
        const bool updated = val == expected;
        expected = val;
        if (updated && success != std::memory_order_relaxed) {
            FENCE_RELEASE;
        } else if (!updated && failure != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return updated;
#else
        //return __atomic_compare_exchange_n(&m_value, &expected, desired, 1, static_cast<int>(success), static_cast<int>(failure));
        return true;
#endif
    }

    __host__ __device__ bool compare_exchange_weak(T& expected, T desired, std::memory_order success, std::memory_order failure) volatile noexcept
    {
        assert(failure != std::memory_order_release);
        assert(failure != std::memory_order_acq_rel);

#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicCAS_system((T*)&m_value, expected, desired);
        const bool updated = val == expected;
        expected = val;
        if (updated && success != std::memory_order_relaxed) {
            FENCE_RELEASE;
        } else if (!updated && failure != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return updated;
#else
        //return __atomic_compare_exchange_n(&m_value, &expected, desired, 1, static_cast<int>(success), static_cast<int>(failure));
        return true;
#endif
    }

    __host__ __device__ bool compare_exchange_weak(T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
        return compare_exchange_weak(expected, desired, order, compare_exchange_duo(order));
    }

    __host__ __device__ bool compare_exchange_weak(T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
        return compare_exchange_weak(expected, desired, order, compare_exchange_duo(order));
    }

    __host__ __device__ bool compare_exchange_strong(T& expected, T desired, std::memory_order success, std::memory_order failure) noexcept
    {
        assert(failure != std::memory_order_release);
        assert(failure != std::memory_order_acq_rel);

#if defined(__CUDA_ARCH__)
        return compare_exchange_weak(expected, desired, success, failure);
#else
        //return __atomic_compare_exchange_n(&m_value, &expected, desired, 0, static_cast<int>(success), static_cast<int>(failure));
        return true;
#endif
    }

    __host__ __device__ bool compare_exchange_strong(T& expected, T desired, std::memory_order success, std::memory_order failure) volatile noexcept
    {
        assert(failure != std::memory_order_release);
        assert(failure != std::memory_order_acq_rel);

#if defined(__CUDA_ARCH__)
        return compare_exchange_weak(expected, desired, success, failure);
#else
        //return __atomic_compare_exchange_n(&m_value, &expected, desired, 0, static_cast<int>(success), static_cast<int>(failure));
        return true;
#endif
    }

    __host__ __device__ bool compare_exchange_strong(T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
        return compare_exchange_strong(expected, desired, order, compare_exchange_duo(order));
    }

    __host__ __device__ bool compare_exchange_strong(T& expected, T desired, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
        return compare_exchange_strong(expected, desired, order, compare_exchange_duo(order));
    }

    __host__ __device__ T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAdd_system(&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_add(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_add(T arg, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAdd_system((T*)&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_add(&m_value, arg, static_cast<int>(order));
        return T{};

#endif
    }

    __host__ __device__ T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicSub_system(&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_sub(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_sub(T arg, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicSub_system((T*)&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_sub(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_and(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAnd_system(&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_and(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_and(T arg, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicAnd_system((T*)&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_and(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicOr_system(&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_or(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_or(T arg, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicOr_system((T*)&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_or(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_xor(T arg, std::memory_order order = std::memory_order_seq_cst) noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicXor_system(&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_xor(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T fetch_xor(T arg, std::memory_order order = std::memory_order_seq_cst) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        if (order != std::memory_order_relaxed) {
            FENCE_ACQUIRE;
        }
        const T val = atomicXor_system((T*)&m_value, arg);
        if (order != std::memory_order_relaxed) {
            FENCE_RELEASE;
        }
        return val;
#else
        //return __atomic_fetch_xor(&m_value, arg, static_cast<int>(order));
        return T{};
#endif
    }

    __host__ __device__ T operator++() noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAdd_system(&m_value, 1);
        FENCE_RELEASE;
        return val + 1;
#else
        //return __atomic_add_fetch(&m_value, 1, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator++() volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAdd_system((T*)&m_value, 1);
        FENCE_RELEASE;
        return val + 1;
#else
        //return __atomic_add_fetch(&m_value, 1, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator++(int) noexcept
    {
        return fetch_add(1);
    }

    __host__ __device__ T operator++(int) volatile noexcept
    {
        return fetch_add(1);
    }

    __host__ __device__ T operator--() noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicSub_system(&m_value, 1);
        FENCE_RELEASE;
        return val - 1;
#else
        //return __atomic_sub_fetch(&m_value, 1, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator--() volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicSub_system((T*)&m_value, 1);
        FENCE_RELEASE;
        return val - 1;
#else
        //return __atomic_sub_fetch(&m_value, 1, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator--(int) noexcept
    {
        return fetch_sub(1);
    }

    __host__ __device__ T operator--(int) volatile noexcept
    {
        return fetch_sub(1);
    }

    __host__ __device__ T operator+=(T arg) noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAdd_system(&m_value, arg);
        FENCE_RELEASE;
        return val + arg;
#else
        //return __atomic_add_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator+=(T arg) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAdd_system((T*)&m_value, arg);
        FENCE_RELEASE;
        return val + arg;
#else
        //return __atomic_add_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator-=(T arg) noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicSub_system(&m_value, arg);
        FENCE_RELEASE;
        return val - arg;
#else
        //return __atomic_sub_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator-=(T arg) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicSub_system((T*)&m_value, arg);
        FENCE_RELEASE;
        return val - arg;
#else
        //return __atomic_sub_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator&=(T arg) noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAnd_system(&m_value, arg);
        FENCE_RELEASE;
        return val & arg;
#else
        //return __atomic_and_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator&=(T arg) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicAnd_system((T*)&m_value, arg);
        FENCE_RELEASE;
        return val & arg;
#else
        //return __atomic_and_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator|=(T arg) noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicOr_system(&m_value, arg);
        FENCE_RELEASE;
        return val | arg;
#else
        //return __atomic_or_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator|=(T arg) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicOr_system((T*)&m_value, arg);
        FENCE_RELEASE;
        return val | arg;
#else
        //return __atomic_or_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator^=(T arg) noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicXor_system(&m_value, arg);
        FENCE_RELEASE;
        return val ^ arg;
#else
        //return __atomic_xor_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

    __host__ __device__ T operator^=(T arg) volatile noexcept
    {
#if defined(__CUDA_ARCH__)
        FENCE_ACQUIRE;
        const T val = atomicXor_system((T*)&m_value, arg);
        FENCE_RELEASE;
        return val ^ arg;
#else
        //return __atomic_xor_fetch(&m_value, arg, __ATOMIC_SEQ_CST);
        return T{};
#endif
    }

private:
    static constexpr int k_alignment = (sizeof(T) > alignof(T)) ? sizeof(T) : alignof(T);
    alignas(k_alignment) T m_value;
};

template <typename T>
struct atomic;

template <>
struct atomic<char> : atomic_int_base<int>
{
    using int_type  = int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<signed char> : atomic_int_base<signed int>
{
    using int_type  = signed int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<unsigned char> : atomic_int_base<unsigned int>
{
    using int_type  = unsigned int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<short> : atomic_int_base<int>
{
    using int_type  = int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<unsigned short> : atomic_int_base<unsigned int>
{
    using int_type  = unsigned int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<int> : atomic_int_base<int>
{
    using int_type  = int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<unsigned int> : atomic_int_base<unsigned int>
{
    using int_type  = unsigned int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<long> : atomic_int_base<int>
{
    using int_type  = int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

template <>
struct atomic<unsigned long> : atomic_int_base<unsigned int>
{
    using int_type  = unsigned int;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};

#if 0
template <>
struct atomic<long long> : atomic_int_base<long long>
{
    using int_type  = long long;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};
#endif

template <>
struct atomic<unsigned long long> : atomic_int_base<unsigned long long>
{
    using int_type  = unsigned long long;
    using base_type = atomic_int_base<int_type>;

    __host__ __device__ constexpr atomic() noexcept : base_type() { }
    __host__ __device__ ~atomic() noexcept { }
    __host__ __device__ atomic(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) = delete;
    __host__ __device__ atomic& operator=(const atomic&) volatile = delete;

    __host__ __device__ constexpr atomic(int_type i) noexcept : base_type(i) { }

    using base_type::operator int_type;
    using base_type::operator=;
};
