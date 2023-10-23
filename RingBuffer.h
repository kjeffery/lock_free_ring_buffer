//
// Created by Keith on 11/24/2020.
//

#pragma once

#include "Finally.h"
#include "Memory.h"
#include "Wait.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>
#include <utility>
#include <bit>

enum class ConsumerTraits
{
    multiple_consumers,
    single_consumer
};

enum class ProducerTraits
{
    multiple_producers,
    single_producer
};

enum class LockTraits
{
    auto_detect,
    spin_lock,
    lock_free
};

struct DefaultRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::multiple_producers;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::multiple_consumers;
    static constexpr LockTraits     lock_traits     = LockTraits::auto_detect;
};

template <typename T>
constexpr bool can_use_type_noexcept()
{
    return std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>;
}

template <typename T, LockTraits input_lock_traits>
struct GetLockTraits
{
    static constexpr LockTraits traits = input_lock_traits;
};

template <typename T>
struct GetLockTraits<T, LockTraits::auto_detect>
{
    static constexpr LockTraits traits = (can_use_type_noexcept<T>()) ? LockTraits::spin_lock : LockTraits::lock_free;
};

#if defined(COUNT_LOOPS)
extern std::atomic<unsigned> loop_iterations;
#endif

template <typename T, std::size_t log_n_elements, typename Traits>
class RingBufferImpl
{
    enum class occupied_t : std::int_least8_t
    {
        k_unoccupied,
        k_in_transition,
        k_occupied,
        k_exception
    };

    using index_t                       = std::uint32_t;
    using atomic_index_t                = std::atomic<index_t>;
    using atomic_occupied_t             = std::atomic<occupied_t>;
    static constexpr index_t k_capacity = 1u << log_n_elements;

    enum class PopImplementation
    {
        move,
        placement
    };

    static constexpr LockTraits k_lock_traits = GetLockTraits<T, Traits::lock_traits>::traits;

public:
    using value_type = T;

    RingBufferImpl() noexcept
    {
        // Check for lock-free types. atomic_occupied_t can easily be changed to an integer type that supports lock-free
        // atomic operations.
        static_assert(atomic_index_t::is_always_lock_free, "Expecting lock-free types");
        static_assert(atomic_occupied_t::is_always_lock_free, "Expecting lock-free types");
    }

    ~RingBufferImpl()
    {
        // Instead of just calling pop() until we're empty, we do it manually to save some atomic operations.
        index_t remaining = size();
        index_t idx       = m_read_idx.load(std::memory_order_relaxed);
        while (remaining > 0) {
            const auto occupied_status = m_nodes[idx].m_occupied.load(std::memory_order_relaxed);
            assert(occupied_status != occupied_t::k_in_transition);
            if (occupied_status == occupied_t::k_occupied) {
                get_pointer(idx)->~T();
                --remaining;
            }
            idx = increment(idx);
        }
    }

    RingBufferImpl(const RingBufferImpl&)            = delete;
    RingBufferImpl(RingBufferImpl&&)                 = delete;
    RingBufferImpl& operator=(const RingBufferImpl&) = delete;
    RingBufferImpl& operator=(RingBufferImpl&&)      = delete;

    static constexpr LockTraits get_lock_traits() noexcept
    {
        return k_lock_traits;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return m_size == 0;
    }

    [[nodiscard]] bool full() const noexcept
    {
        return m_size == k_capacity;
    }

    [[nodiscard]] index_t size() const noexcept
    {
        return m_size;
    }

    static constexpr index_t capacity() noexcept
    {
        return k_capacity;
    }

    [[nodiscard]] bool try_push(const T& t)
    {
        return try_push_impl(t);
    }

    [[nodiscard]] bool try_push(T&& t)
    {
        return try_push_impl(std::move(t));
    }

    void push(const T& t)
    {
        push_impl(t);
    }

    void push(T&& t)
    {
        push_impl(std::move(t));
    }

    template <typename Iterator>
    [[nodiscard]] bool try_push_batch(Iterator first, Iterator last)
    {
        static_assert(Traits::producer_traits == ProducerTraits::single_producer,
                      "Batch mode supported for single producer only");
        index_t throwaway;
        return try_push_batch_impl(first, last, throwaway);
    }

    template <typename Iterator>
    void push_batch(Iterator first, Iterator last)
    {
        push_batch(first, last, typename std::iterator_traits<Iterator>::iterator_category());
    }

    template <typename... Args>
    void emplace(Args&&... args)
    {
        constexpr bool no_except_construct = noexcept(T(std::forward<Args>(args)...));
        if constexpr (k_lock_traits == LockTraits::spin_lock && !no_except_construct) {
            // We don't have a way to check which constructor is called by emplace for our class static exception
            // checks used to determine our lock traits. We only check exception safety in move construction and move
            // assignment when we check our top-level traits. Check here, and if it can throw, create our type before we
            // call push in an effort to use the move operations.
            push_impl(T(std::forward<Args>(args)...));
        } else {
            push_impl(std::forward<Args>(args)...);
        }
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    [[nodiscard]] bool try_pop(T& ret)
    {
        using enum ConsumerTraits;
        using enum LockTraits;

        if constexpr (Traits::consumer_traits == single_consumer && k_lock_traits == spin_lock) {
            return try_pop_single_impl<PopImplementation::move>(ret);
        } else if constexpr (Traits::consumer_traits == single_consumer && k_lock_traits == lock_free) {
            return try_pop_single_lock_free_impl<PopImplementation::move>(ret);
        } else if constexpr (Traits::consumer_traits == multiple_consumers && k_lock_traits == spin_lock) {
            return try_pop_impl<PopImplementation::move>(ret);
        } else {
            return try_pop_lock_free_impl<PopImplementation::move>(ret);
        }
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    T pop()
    {
        // We don't want to make a default constructor a requirement of T, so we just allocate stack space for it, and
        // tell our pop function to do placement new for a copy constructor.
        storage_t  storage;
        const auto p = std::bit_cast<T*>(std::addressof(storage));
        do_pop_dispatch<PopImplementation::placement>(*p);
        T ret{std::move(*p)};
        p->~T();
        return ret;
    }

    // This method does not meet the strong exception guarantee.
    // If T's move constructor on return throws, we've lost that data forever.
    void pop(T& ret)
    {
        do_pop_dispatch<PopImplementation::move>(ret);
    }

private:
    template <typename... Args>
    void do_construct(index_t write_idx, Args&&... args)
    {
        new (get_pointer(write_idx)) T(std::forward<Args>(args)...);
    }

    // Sometimes our forwarded arguments are actually the type itself (as opposed to arguments for creating the type).
    // Do a move if we can, but if we throw, we don't want to lose the data the user is passing in.
    void do_construct(index_t write_idx, T&& t)
    {
        new (get_pointer(write_idx)) T(std::move_if_noexcept(t));
    }

    template <PopImplementation pop_implementation>
    void do_pop_dispatch(T& ret)
    {
        using enum ConsumerTraits;
        using enum LockTraits;

        // We don't want to make a default constructor a requirement of T, so we just allocate stack space for it, and
        // tell our pop function to do placement new for a copy constructor.
        bool popped;
        do {
            if constexpr (Traits::consumer_traits == single_consumer && k_lock_traits == spin_lock) {
                popped = try_pop_single_impl<pop_implementation>(ret);
            } else if constexpr (Traits::consumer_traits == single_consumer && k_lock_traits == lock_free) {
                popped = try_pop_single_lock_free_impl<pop_implementation>(ret);
            } else if constexpr (Traits::consumer_traits == multiple_consumers && k_lock_traits == spin_lock) {
                popped = try_pop_impl<pop_implementation>(ret);
            } else {
                popped = try_pop_lock_free_impl<pop_implementation>(ret);
            }
            if (!popped) {
                m_size.wait(0u);
            }
        } while (!popped);
    }

    template <typename... Args>
    void push_impl(Args&&... args)
    {
        using enum LockTraits;
        using enum ProducerTraits;

        bool pushed;
        do {
            // Does it make you nervous to see std::forward in a loop? I understand. The contents, however, will not be
            // consumed unless the push succeeds (i.e. there will be no moves until success).
            if constexpr (Traits::producer_traits == single_producer && k_lock_traits == spin_lock) {
                pushed = try_push_single_impl(std::forward<Args>(args)...);
            } else if constexpr (Traits::producer_traits == single_producer && k_lock_traits == lock_free) {
                pushed = try_push_single_lock_free_impl(std::forward<Args>(args)...);
            } else if constexpr (Traits::producer_traits == multiple_producers && k_lock_traits == spin_lock) {
                pushed = try_push_impl(std::forward<Args>(args)...);
            } else {
                pushed = try_push_lock_free_impl(std::forward<Args>(args)...);
            }
            if (!pushed) {
                m_size.wait(k_capacity);
            }
        } while (!pushed);
    }

    // This version of push assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <typename... Args>
    [[nodiscard]] bool try_push_impl(Args&&... t)
    {
        index_t size = m_size;
        if (size == k_capacity) {
            return false;
        }
        // Increment size counter. This allows us to reserve our space before any changes take place.
        // A failed CAS may set size to capacity, so check for a full buffer again in loop.
        while (!m_size.compare_exchange_weak(size, size + 1u)) {
            if (size == k_capacity) {
                return false;
            }
        }

        // We know we have space (and have reserved it) at this point. We just have to grab our write index.

        // Get and increment write pos.
        index_t write_idx = m_write_idx;
        while (!m_write_idx.compare_exchange_weak(write_idx, increment(write_idx)))
            ;

        assert(write_idx < k_capacity);

        // Wait until our slot isn't occupied, and then mark it as in transition so that:
        // A) other writers know they can't use this index.
        // B) readers know they can't read from this index yet.
        //
        // What guarantees that our slot doesn't get stolen by another writer between obtaining our index and marking it
        // as in transition? Nothing. It's completely possible that we have wrapped around the buffer and that a later
        // thread is writing to our slot, but they will have to mark the slot as in transition or occupied, stopping us
        // from writing to it at the same point, and it will become available once a reader clears it up.
        //
        // What guarantees us that a reader will read from our slot if another writer has stolen it? The read index will
        // have to be incremented. If we have looped so that there are two writes, we will eventually loop so that there
        // are two reads.
        occupied_t occupied_value = occupied_t::k_unoccupied;
        while (!m_nodes[write_idx].m_occupied.compare_exchange_weak(occupied_value, occupied_t::k_in_transition)) {
            // We only want to update this when it's unoccupied, so reset our expected value.
            m_nodes[write_idx].m_occupied.wait(occupied_value);
            occupied_value = occupied_t::k_unoccupied;
        }

        // Construct
        do_construct(write_idx, std::forward<Args>(t)...);

        // Inform readers that this slot if valid and can be read from.
        m_nodes[write_idx].m_occupied = occupied_t::k_occupied;
        // We may notify another writer after we've filled it up again, which does us no good.
        m_size.notify_all();
        m_nodes[write_idx].m_occupied.notify_all();
        return true;
    }

    // This version of push handles the class throwing exceptions while still maintaining lock-free semantics.
    template <typename... Args>
    [[nodiscard]] bool try_push_lock_free_impl(Args&&... t)
    {
        index_t size = m_size;
        if (size == k_capacity) {
            return false;
        }
        // Increment size counter. This allows us to reserve our space before any changes take place.
        // A failed CAS may set size to capacity, so check for a full buffer again in loop.
        while (!m_size.compare_exchange_weak(size, size + 1u)) {
            if (size == k_capacity) {
                return false;
            }
        }

        // We know we have space (and have reserved it) at this point. We just have to grab our write index.

        while (true) {
            // Get and increment write pos.
            index_t write_idx = m_write_idx;
            while (!m_write_idx.compare_exchange_weak(write_idx, increment(write_idx)))
                ;

            assert(write_idx < k_capacity);

            // Wait until our slot isn't occupied, and then mark it as in transition so that:
            // A) other writers know they can't use this index.
            // B) readers know they can't read from this index yet.
            //
            // What guarantees that our slot doesn't get stolen by another writer between obtaining our index and
            // marking it as in transition? Nothing. It's completely possible that we have wrapped around the buffer and
            // that a later thread is writing to our slot, but they will have to mark the slot as in transition or
            // occupied, stopping us from writing to it at the same point, and it will become available once a reader
            // clears it up.
            //
            // What guarantees us that a reader will read from our slot if another writer has stolen it? The read index
            // will have to be incremented. If we have looped so that there are two writes, we will eventually loop so
            // that there are two reads.
            occupied_t occupied_value = occupied_t::k_unoccupied;
            if (m_nodes[write_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::k_in_transition)) {
                auto cleanup = finally([this, write_idx] {
                    // We may notify another writer after we've filled it up again, which does us no good, so we notify
                    // all threads.
                    m_size.notify_all();
                    m_nodes[write_idx].m_occupied.notify_all();
                });

                try {
                    // Construct
                    do_construct(write_idx, std::forward<Args>(t)...);
                } catch (...) {
                    m_nodes[write_idx].m_occupied = occupied_t::k_exception;
                    throw;
                }

                // Inform readers that this slot if valid and can be read from.
                m_nodes[write_idx].m_occupied = occupied_t::k_occupied;
                return true;
            }
        }
    }

    // Special case for a single producer.
    // This version of push assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <typename... Args>
    [[nodiscard]] bool try_push_single_impl(Args&&... t)
    {
        // We're the only writer. m_size will only decrease.
        if (m_size == k_capacity) {
            return false;
        }

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for size or for the data to be ready.
        ++m_size;

        // Get and increment write pos.
        const index_t write_idx =
            m_write_idx.exchange(increment(m_write_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
        assert(write_idx < k_capacity);

        // Wait until it's something other than occupied.
        // Can't atomic::wait here because we need to wait if we're either occupied or in transition
        while (m_nodes[write_idx].m_occupied != occupied_t::k_unoccupied) {
            do_pause();
        }

        // Construct
        do_construct(write_idx, std::forward<Args>(t)...);

        // Inform readers that this slot if valid and can be read from.
        m_nodes[write_idx].m_occupied = occupied_t::k_occupied;

        // We only need to wake up one thread, but we have two wait variables, and we can't control which thread gets
        // the notification. Inform one thread that the size has changed, and inform all threads (including the one that
        // knows the size has changed) that the status has changed
        m_size.notify_one();
        m_nodes[write_idx].m_occupied.notify_all();
        return true;
    }

    // Special case for a single producer.
    // This version of push handles the class throwing exceptions while still maintaining lock-free semantics.
    template <typename... Args>
    [[nodiscard]] bool try_push_single_lock_free_impl(Args&&... t)
    {
        // We're the only writer. m_size will only decrease.
        if (m_size == k_capacity) {
            return false;
        }

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for size or for the data to be ready.
        ++m_size;

        while (true) {
            // Get and increment write pos.
            const index_t write_idx =
                m_write_idx.exchange(increment(m_write_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
            assert(write_idx < k_capacity);

            occupied_t occupied_value = occupied_t::k_unoccupied;
            if (m_nodes[write_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::k_in_transition)) {
                auto cleanup = finally([this, write_idx] {
                    // There are only other readers, so we don't need to worry about waking another writer.
                    // We only need to wake up one thread, but we have two wait variables, and we can't control which
                    // thread gets the notification. Inform one thread that the size has changed, and inform all threads
                    // (including the one that knows the size has changed) that the status has changed
                    m_size.notify_one();
                    m_nodes[write_idx].m_occupied.notify_all();
                });

                try {
                    // Construct
                    do_construct(write_idx, std::forward<Args>(t)...);
                } catch (...) {
                    m_nodes[write_idx].m_occupied = occupied_t::k_exception;
                    throw;
                }

                // Inform readers that this slot if valid and can be read from.
                m_nodes[write_idx].m_occupied = occupied_t::k_occupied;
                return true;
            }
        }
    }

    template <typename Iterator>
    void push_batch(Iterator first, Iterator last, std::input_iterator_tag)
    {
        static_assert(Traits::producer_traits == ProducerTraits::single_producer,
                      "Batch mode supported for single producer only");

        // With input iterators, we can't count, and we can't go over the values again. The best we can do is add them
        // one at a time.
        for (; first != last; ++first) {
            push_impl(*first);
        }
    }

    // This function works for forward iterators, but is more efficient with random access iterators.
    template <typename Iterator>
    void push_batch(Iterator first, Iterator last, std::forward_iterator_tag)
    {
        static_assert(Traits::producer_traits == ProducerTraits::single_producer,
                      "Batch mode supported for single producer only");
        index_t num_elements_to_process = std::distance(first, last);

        while (first != last) {
            // We're the only writer. m_size will only decrease, meaning this is pessimistic (which is perfect).
            const auto container_size = m_size.load(std::memory_order_acquire);
            if (container_size == capacity()) {
                m_size.wait(container_size);
            }

            const index_t mini_batch_size = std::min(num_elements_to_process, capacity() - container_size);

            assert(mini_batch_size <= static_cast<index_t>(std::distance(first, last)));
            const auto           next    = std::next(first, mini_batch_size);
            [[maybe_unused]] bool success = try_push_batch_impl(first, next, mini_batch_size);
            assert(success);
            first = next;
            num_elements_to_process -= mini_batch_size;
        }
    }

    // This version uses a(n implicit) spinlock.
    // We pass in num_elements even though it should be the difference between first and last so that we don't have to
    // iterate over [first, last] when using forward_iterators.
    // Precondition: This function should only be called when there is enough room to add num_elements.
    template <typename Iterator>
    [[nodiscard]] bool try_push_batch_impl(Iterator first, Iterator last, index_t num_elements)
    {
        // TODO: C++20: This is a good case for concepts.
        static_assert(std::is_base_of<std::forward_iterator_tag,
                          typename std::iterator_traits<Iterator>::iterator_category>::value,
                      "This only works for forward iterators or better");

        // We pass in num_elements even though it can be derived through the iterators for efficiency in the case of
        // forward iterators.
        assert(std::distance(first, last) == num_elements);

        // Since we're the one producer, this should succeed if this function is called properly.
        assert(capacity() - m_size.load() >= num_elements);

        // We only have one thread inserting. We don't have to increment this right away, but either way, asynchronous
        // readers will be busy waiting either for container_size or for the data to be ready. Doing it here means:
        // * We only pay for this atomic operation once.
        // * Consumers can get data as soon as we update the occupied state.
        // The drawback is that we block consumers (even if they do a try_pop) because they think there is data to be
        // read but the data is inaccessible.
        m_size += num_elements;
        notify_all(m_size);

        index_t write_idx = m_write_idx.load(std::memory_order_relaxed);
        for (; first != last; ++first, write_idx = increment(write_idx)) {
            assert(write_idx < k_capacity);

            // Wait until it's something other than occupied.
            // Can't atomic::wait here because we need to wait if we're either unoccupied or in transition
            while (m_nodes[write_idx].m_occupied != occupied_t::k_unoccupied) {
                do_pause();
            }

            // Construct
            new (get_pointer(write_idx)) T(std::move_if_noexcept(*first));

            // Inform readers that this slot is valid and can be read from.
            m_nodes[write_idx].m_occupied = occupied_t::k_occupied;

            // We have already informed all threads about the size change. We won't wake another writer (we're the only
            // one), so just wake one thread on the status change.
            m_nodes[write_idx].m_occupied.notify_one();
        }
        m_write_idx.store(write_idx, std::memory_order_relaxed);

        // If uploading memory, do memcpy and do a m_size.notify_all() here instead of the notify one

        return true;
    }

    // This version of pop assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <PopImplementation pop_implementation>
    [[nodiscard]] bool try_pop_impl(T& ret)
    {
        // Check for empty buffer
        // Decrement counter
        index_t size = m_size;
        if (size == 0) {
            return false;
        }
        while (!m_size.compare_exchange_weak(size, size - 1u)) {
            if (size == 0) {
                return false;
            }
        }

        assert(size <= k_capacity);

        // Get and increment read pos
        index_t read_idx = m_read_idx;
        while (!m_read_idx.compare_exchange_weak(read_idx, increment(read_idx)))
            ;

        assert(read_idx < k_capacity);

        // Wait until we have valid data
        occupied_t occupied_value = occupied_t::k_occupied;
        while (!m_nodes[read_idx].m_occupied.compare_exchange_weak(occupied_value, occupied_t::k_in_transition)) {
            m_nodes[read_idx].m_occupied.wait(occupied_value);
            occupied_value = occupied_t::k_occupied;
        }

        try {
            if constexpr (pop_implementation == PopImplementation::placement) {
                new (std::addressof(ret)) T{std::move_if_noexcept(*get_pointer(read_idx))};
            } else if constexpr (pop_implementation == PopImplementation::move) {
                ret = std::move_if_noexcept(*get_pointer(read_idx));
            }

            // Destroy
            get_pointer(read_idx)->~T();

            // Mark as not occupied
            m_nodes[read_idx].m_occupied = occupied_t::k_unoccupied;
            // We may notify another reader after we've emptied it up again, which does us no good.
            m_size.notify_all();
            m_nodes[read_idx].m_occupied.notify_all();
        } catch (...) {
            get_pointer(read_idx)->~T();
            m_nodes[read_idx].m_occupied = occupied_t::k_occupied;
            // We may notify another reader after we've emptied it up again, which does us no good.
            m_size.notify_all();
            m_nodes[read_idx].m_occupied.notify_all();
            throw;
        }
        return true;
    }

    // This version of pop handles the class throwing exceptions while still maintaining lock-free semantics.
    template <PopImplementation pop_implementation>
    [[nodiscard]] bool try_pop_lock_free_impl(T& ret)
    {
        // Check for empty buffer
        // Decrement counter
        index_t size = m_size;
        if (size == 0) {
            return false;
        }
        while (!m_size.compare_exchange_weak(size, size - 1u)) {
            if (size == 0) {
                return false;
            }
        }

        assert(size <= k_capacity);

        while (true) {
#if defined(COUNT_LOOPS)
            ++loop_iterations;
#endif
            // Get and increment read pos
            index_t read_idx = m_read_idx;
            while (!m_read_idx.compare_exchange_weak(read_idx, increment(read_idx)))
                ;

            assert(read_idx < k_capacity);

            auto cleanup_function = [this, read_idx] {
                // Mark as not occupied
                m_nodes[read_idx].m_occupied = occupied_t::k_unoccupied;

                // We may notify another reader after we've emptied it up again, which does us no good.
                m_size.notify_all();
                m_nodes[read_idx].m_occupied.notify_all();
            };

            // Wait until we have valid data
            occupied_t occupied_value = occupied_t::k_occupied;
            if (m_nodes[read_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::k_in_transition)) {
                auto cleanup = finally(cleanup_function);

                try {
                    if constexpr (pop_implementation == PopImplementation::placement) {
                        new (std::addressof(ret)) T{std::move_if_noexcept(*get_pointer(read_idx))};
                    } else if constexpr (pop_implementation == PopImplementation::move) {
                        ret = std::move_if_noexcept(*get_pointer(read_idx));
                    }

                    // Destroy
                    get_pointer(read_idx)->~T();
                    return true;
                } catch (...) {
                    get_pointer(read_idx)->~T();
                    throw;
                }
            } else if (occupied_value == occupied_t::k_exception) {
                auto cleanup = finally(cleanup_function);
                return false;
            }
        }
    }

    // Special case for a single consumer.
    // This version of pop assumes that the operations on the stored class will not throw exceptions, and that nothing
    // will interfere with the thread executing the function to completion.
    // This version uses a(n implicit) spinlock.
    template <PopImplementation pop_implementation>
    [[nodiscard]] bool try_pop_single_impl(T& ret)
    {
        // Check for empty buffer
        // We're the only reader. m_size will only increase.
        if (m_size == 0) {
            return false;
        }

        // We only have one thread reading. We don't have to decrement this right away, but either way, asynchronous
        // writers will be busy waiting either for size or for the data to be ready.
        --m_size;

        // Get and increment read pos
        const index_t read_idx =
            m_read_idx.exchange(increment(m_read_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
        assert(read_idx < k_capacity);

        // Wait until we have valid data
        // Can't atomic::wait here because we need to wait if we're either unoccupied or in transition
        while (m_nodes[read_idx].m_occupied != occupied_t::k_occupied) {
            do_pause();
        }

        try {
            if constexpr (pop_implementation == PopImplementation::placement) {
                new (std::addressof(ret)) T{std::move_if_noexcept(*get_pointer(read_idx))};
            } else if constexpr (pop_implementation == PopImplementation::move) {
                ret = std::move_if_noexcept(*get_pointer(read_idx));
            }

            // Destroy
            get_pointer(read_idx)->~T();

            // Mark as not occupied
            m_nodes[read_idx].m_occupied = occupied_t::k_unoccupied;

            // We only need to wake up one thread, but we have two wait variables, and we can't control which
            // thread gets the notification. Inform one thread that the size has changed, and inform all threads
            // (including the one that knows the size has changed) that the status has changed
            m_size.notify_one();
            m_nodes[read_idx].m_occupied.notify_all();
        } catch (...) {
            get_pointer(read_idx)->~T();
            m_nodes[read_idx].m_occupied = occupied_t::k_occupied;

            // We only need to wake up one thread, but we have two wait variables, and we can't control which
            // thread gets the notification. Inform one thread that the size has changed, and inform all threads
            // (including the one that knows the size has changed) that the status has changed
            m_size.notify_one();
            m_nodes[read_idx].m_occupied.notify_all();
            throw;
        }
        return true;
    }

    // Special case for a single consumer.
    // This version of pop handles the class throwing exceptions while still maintaining lock-free semantics.
    template <PopImplementation pop_implementation>
    [[nodiscard]] bool try_pop_single_lock_free_impl(T& ret)
    {
        // Check for empty buffer
        // We're the only reader. m_size will only increase.
        if (m_size == 0) {
            return false;
        }

        // We only have one thread reading. We don't have to decrement this right away, but either way, asynchronous
        // writers will be busy waiting either for size or for the data to be ready.
        --m_size;

        while (true) {
            // Get and increment read pos
            const index_t read_idx =
                m_read_idx.exchange(increment(m_read_idx.load(std::memory_order_relaxed)), std::memory_order_relaxed);
            assert(read_idx < k_capacity);

            auto cleanup_function = [this, read_idx] {
                // Mark as not occupied
                m_nodes[read_idx].m_occupied = occupied_t::k_unoccupied;

                // There are only other writers, so we don't need to worry about waking another reader.
                // We only need to wake up one thread, but we have two wait variables, and we can't control which
                // thread gets the notification. Inform one thread that the size has changed, and inform all threads
                // (including the one that knows the size has changed) that the status has changed
                m_size.notify_one();
                m_nodes[read_idx].m_occupied.notify_all();
            };

            // Check to see if we have valid data
            occupied_t occupied_value = occupied_t::k_occupied;
            if (m_nodes[read_idx].m_occupied.compare_exchange_strong(occupied_value, occupied_t::k_in_transition)) {
                auto cleanup = finally(cleanup_function);

                try {
                    if constexpr (pop_implementation == PopImplementation::placement) {
                        new (std::addressof(ret)) T{std::move_if_noexcept(*get_pointer(read_idx))};
                    } else if constexpr (pop_implementation == PopImplementation::move) {
                        ret = std::move_if_noexcept(*get_pointer(read_idx));
                    }

                    // Destroy
                    get_pointer(read_idx)->~T();
                    return true;
                } catch (...) {
                    get_pointer(read_idx)->~T();
                    throw;
                }
            } else if (occupied_value == occupied_t::k_exception) {
                auto cleanup = finally(cleanup_function);
                return false;
            }
        }
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
        return std::bit_cast<T*>(&m_nodes[i].m_storage);
    }

    const T* get_pointer(index_t i) const noexcept
    {
        return std::bit_cast<T*>(&m_nodes[i].m_storage);
    }

    using storage_t = std::array<std::byte, sizeof(T)>;

    struct alignas(std::min<std::size_t>(16u, alignof(T))) Node
    {
        alignas(alignof(T)) storage_t m_storage;
        atomic_occupied_t m_occupied{occupied_t::k_unoccupied};
    };

    alignas(k_cache_line_size) atomic_index_t m_size{0};
    alignas(k_cache_line_size) atomic_index_t m_read_idx{0};
    alignas(k_cache_line_size) atomic_index_t m_write_idx{0};
    alignas(k_cache_line_size) std::unique_ptr<Node[]> m_nodes{new Node[k_capacity]};
};

struct SingleProducerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::single_producer;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::multiple_consumers;
    static constexpr LockTraits     lock_traits     = LockTraits::auto_detect;
};

struct SingleConsumerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::multiple_producers;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::single_consumer;
    static constexpr LockTraits     lock_traits     = LockTraits::auto_detect;
};

struct SingleProducerSingleConsumerRingBufferTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::single_producer;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::single_consumer;
    static constexpr LockTraits     lock_traits     = LockTraits::auto_detect;
};

struct LockFreeTraits
{
    static constexpr ProducerTraits producer_traits = ProducerTraits::multiple_producers;
    static constexpr ConsumerTraits consumer_traits = ConsumerTraits::multiple_consumers;
    static constexpr LockTraits     lock_traits     = LockTraits::lock_free;
};

template <typename T, std::size_t log_n_elements>
using RingBuffer = RingBufferImpl<T, log_n_elements, DefaultRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleProducer = RingBufferImpl<T, log_n_elements, SingleProducerRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleConsumer = RingBufferImpl<T, log_n_elements, SingleConsumerRingBufferTraits>;

template <typename T, std::size_t log_n_elements>
using RingBufferSingleProducerSingleConsumer =
    RingBufferImpl<T, log_n_elements, SingleProducerSingleConsumerRingBufferTraits>;
