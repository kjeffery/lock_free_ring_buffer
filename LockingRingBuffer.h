//
// Created by Keith on 2/15/2021.
//

#pragma once

#include "Memory.h"

#include <condition_variable>
#include <cstdint>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

template <typename T, std::size_t log_n_elements>
class LockingRingBuffer
{
    using index_t                       = uint32_t;
    static constexpr index_t k_capacity = 1u << log_n_elements;

public:
    LockingRingBuffer() noexcept
    : m_read_idx(0)
    , m_write_idx(0)
    , m_storage(static_cast<storage_t*>(aligned_allocate(sizeof(storage_t) * k_capacity, get_page_size())))
    {
    }

    ~LockingRingBuffer()
    {
        while (!empty()) {
            const index_t read_idx = m_read_idx;
            get_pointer(read_idx)->~T();
            m_read_idx = increment(read_idx);
        }
        aligned_deallocate(m_storage);
    }

    bool empty() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_write_idx == m_read_idx;
    }

    bool full() const noexcept
    {
        index_t write_idx;
        index_t read_idx;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            write_idx = m_write_idx;
            read_idx  = m_read_idx;
        }
        return increment(write_idx) == read_idx;
    }

    index_t size() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (m_write_idx >= m_read_idx) ? m_write_idx - m_read_idx : k_capacity - m_read_idx - m_write_idx;
    }

    static constexpr index_t capacity() noexcept
    {
        return k_capacity;
    }

    void push(const T& t)
    {
        push_impl(t);
    }

    void push(T&& t)
    {
        push_impl(std::move(t));
    }

    void push_single(const T& t)
    {
        push_single_impl(t);
    }

    void push_single(T&& t)
    {
        push_single_impl(std::move(t));
    }

    template <typename... Args>
    void emplace(Args&&... args)
    {
        push_impl(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void emplace_single(Args&&... args)
    {
        push_single_impl(std::forward<Args>(args)...);
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    T pop()
    {
        // Construct
        try {
            storage_t r;
            T* const t = reinterpret_cast<T*>(std::addressof(r));
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond_empty.wait(lock, [this]() { return !empty_no_lock(); });

                new (t) T{std::move_if_noexcept(*get_pointer(m_read_idx))};

                // Destroy
                get_pointer(m_read_idx)->~T();
                m_read_idx = increment(m_read_idx);
            }

            m_cond_full.notify_one();
            return *t;
        } catch (...) {
            // Strong exception semantics here. If the copy constructor throws, we can just try again later.
            m_cond_full.notify_one();
            throw;
        }
    }

    [[nodiscard]] bool try_pop(T& ret)
    {
        // Construct
        try {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (empty_no_lock()) {
                    return false;
                }

                new (std::addressof(ret)) T{std::move_if_noexcept(*get_pointer(m_read_idx))};

                // Destroy
                get_pointer(m_read_idx)->~T();
                m_read_idx = increment(m_read_idx);
            }

            m_cond_full.notify_one();
            return true;
        } catch (...) {
            // Strong exception semantics here. If the copy constructor throws, we can just try again later.
            m_cond_full.notify_one();
            throw;
        }
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    T pop_single()
    {
        // Construct
        try {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond_empty.wait(lock, [this]() { return !empty_no_lock(); });
            }

            const T r = *get_pointer(m_read_idx);

            // Destroy
            get_pointer(m_read_idx)->~T();
            m_read_idx = increment(m_read_idx);

            // unlock
            m_cond_full.notify_one();
            return r;
        } catch (...) {
            // Strong exception semantics here. If the copy constructor throws, we can just try again later.
            m_cond_full.notify_one();
            throw;
        }
    }

private:
    template <typename... Args>
    void push_impl(Args&&... t)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond_full.wait(lock, [this]() { return !full_no_lock(); });

            // Construct
            new (get_pointer(m_write_idx)) T(std::forward<Args>(t)...);
            m_write_idx = increment(m_write_idx);
        }
        m_cond_empty.notify_one();
    }

    // Special case for a single producer.
    template <typename... Args>
    void push_single_impl(Args&&... t)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond_full.wait(lock, [this]() { return !full_no_lock(); });
        }

        // Construct
        new (get_pointer(m_write_idx)) T(std::forward<Args>(t)...);
        m_write_idx = increment(m_write_idx);
        m_cond_empty.notify_one();
    }

    bool empty_no_lock() const noexcept
    {
        return m_write_idx == m_read_idx;
    }

    bool full_no_lock() const noexcept
    {
        return increment(m_write_idx) == m_read_idx;
    }

    index_t size_no_lock() const noexcept
    {
        return (m_write_idx >= m_read_idx) ? m_write_idx - m_read_idx : k_capacity - m_read_idx - m_write_idx;
    }

    static index_t mod(index_t v) noexcept
    {
        return v % k_capacity;
    }

    static index_t increment(index_t v) noexcept
    {
        return mod(v + 1u);
    }

    T* get_pointer(index_t i) noexcept
    {
        return reinterpret_cast<T*>(&m_storage[i]);
    }

    const T* get_pointer(index_t i) const noexcept
    {
        return reinterpret_cast<T*>(&m_storage[i]);
    }

    using storage_t = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    alignas(k_cache_line_size) index_t                      m_read_idx;
    alignas(k_cache_line_size) index_t                      m_write_idx;
    alignas(k_cache_line_size) mutable std::mutex           m_mutex;
    alignas(k_cache_line_size) std::condition_variable      m_cond_empty;
    alignas(k_cache_line_size) std::condition_variable      m_cond_full;
    alignas(k_cache_line_size) storage_t*                   m_storage;
};

