#include "LockingRingBuffer.h"
#include "RingBuffer.h"
#include "StopWatch.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#define PROFILE_RING_BUFFER 1
// #define PROFILE_LOCK_FREE 1
// #define PROFILE_LOCKING_RINGBUFFER 1
// #define PROFILE_TBB_QUEUE 1
// #define PROFILE_TBB_BOUNDED_QUEUE 1
// #define EXPLICIT_TBB_BOUNDED_QUEUE_SIZE 1

#if defined(PROFILE_TBB_BOUNDED_QUEUE) || defined(PROFILE_TBB_QUEUE)
#define USE_TBB_QUEUE 1
#endif

#if defined(USE_TBB_QUEUE)
#include <tbb/concurrent_queue.h>
#endif

using namespace std::chrono_literals;

constexpr std::size_t k_log_queue_size_test    = 10;
constexpr std::size_t k_log_queue_size_profile = 10;
constexpr bool        k_verbose                = true;
constexpr int         k_nvalues                = 1 << 18;
constexpr double      p_throw_construct        = 0.0002;
constexpr double      p_throw_copy_constructor = 0.0000; // Leave at zero: we're not exception safe on return
constexpr double      p_throw_copy_assignment  = 0.0002;
constexpr double      p_throw_move_constructor = 0.0000;
constexpr double      p_throw_move_assignment  = 0.0002;
constexpr int         k_profile_count          = 64;

namespace timing {
using representation_t = long double;
using clock_t          = std::chrono::steady_clock;
using time_point_t     = std::chrono::time_point<clock_t>;
using duration_t       = std::chrono::duration<representation_t>;
} // namespace timing

template <typename T>
const char* get_ratio_name() noexcept
{
    if constexpr (std::ratio_equal<T, std::atto>::value) {
        return "atto";
    }
    if constexpr (std::ratio_equal<T, std::femto>::value) {
        return "femto";
    }
    if constexpr (std::ratio_equal<T, std::pico>::value) {
        return "pico";
    }
    if constexpr (std::ratio_equal<T, std::nano>::value) {
        return "nano";
    }
    if constexpr (std::ratio_equal<T, std::micro>::value) {
        return "micro";
    }
    if constexpr (std::ratio_equal<T, std::milli>::value) {
        return "milli";
    }
    if constexpr (std::ratio_equal<T, std::centi>::value) {
        return "centi";
    }
    if constexpr (std::ratio_equal<T, std::deci>::value) {
        return "deci";
    }
    if constexpr (std::ratio_equal<T, std::deca>::value) {
        return "deca";
    }
    if constexpr (std::ratio_equal<T, std::hecto>::value) {
        return "hecto";
    }
    if constexpr (std::ratio_equal<T, std::kilo>::value) {
        return "kilo";
    }
    if constexpr (std::ratio_equal<T, std::mega>::value) {
        return "mega";
    }
    if constexpr (std::ratio_equal<T, std::giga>::value) {
        return "giga";
    }
    if constexpr (std::ratio_equal<T, std::tera>::value) {
        return "tera";
    }
    if constexpr (std::ratio_equal<T, std::peta>::value) {
        return "peta";
    }
    if constexpr (std::ratio_equal<T, std::exa>::value) {
        return "exa";
    }

    return "Unknown";
}

std::string get_stopwatch_units()
{
    return get_ratio_name<StopWatch::period>() + std::string("seconds");
}

class TestException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class TimingStats
{
    std::atomic<timing::duration_t> m_sum{timing::duration_t::zero()};
    std::atomic<unsigned>           m_count{0};

public:
    TimingStats() = default;

    void reset()
    {
        m_sum   = timing::duration_t::zero();
        m_count = 0;
    }

    void push(timing::duration_t dur)
    {
        auto s = m_sum.load();
        while (!m_sum.compare_exchange_weak(s, s + dur))
            ;
        m_count.fetch_add(1, std::memory_order_relaxed);
    }

    timing::representation_t mean() const
    {
        return m_sum.load().count() / m_count.load();
    }
};

TimingStats g_stats;

template <typename T>
void print_safe_helper(std::ostream& outs, T&& t)
{
    outs << std::forward<T>(t);
}

template <typename T, typename... Args>
void print_safe_helper(std::ostream& outs, T&& t, Args&&... args)
{
    outs << std::forward<T>(t);
    print_safe_helper(outs, std::forward<Args>(args)...);
}

template <typename... Args>
void print_safe(Args&&... args)
{
    std::ostringstream outs;
    print_safe_helper(outs, std::forward<Args>(args)...);
    // printf is, surprisingly, thread safe.
    std::printf("%s\n", outs.str().c_str());
}

void render(std::uint16_t nx, std::uint16_t ny);

std::ostream& print_lock_traits(std::ostream& outs, LockTraits traits)
{
    switch (traits) {
    case LockTraits::auto_detect:
        outs << "auto-detect\n";
        break;
    case LockTraits::spin_lock:
        outs << "spin-lock\n";
        break;
    case LockTraits::lock_free:
        outs << "lock-free\n";
        break;
    }
    return outs;
}

template <typename Traits>
std::ostream& print_traits_info(std::ostream& outs)
{
#if 0
    switch (T::communication_traits) {
        case CommunicationTraits::cpu_to_cpu:
            outs << "CPU to CPU";
            break;
        case CommunicationTraits::cpu_to_gpu:
            outs << "CPU to GPU";
            break;
        case CommunicationTraits::gpu_to_gpu:
            outs << "GPU to GPU";
            break;
        case CommunicationTraits::gpu_to_cpu:
            outs << "GPU to CPU";
            break;
    }
    outs << " with ";
#endif
    switch (Traits::producer_traits) {
    case ProducerTraits::multiple_producers:
        outs << "multiple producer support";
        break;
    case ProducerTraits::single_producer:
        outs << "single producer support";
        break;
    }
    outs << " and ";
    switch (Traits::consumer_traits) {
    case ConsumerTraits::multiple_consumers:
        outs << "multiple consumer support";
        break;
    case ConsumerTraits::single_consumer:
        outs << "single consumer support";
        break;
    }
    outs << " with ";
    print_lock_traits(outs, Traits::lock_traits);
    outs << std::endl;
    return outs;
}

template <typename Traits>
void verify_preconditions(int nproducers, int nconsumers)
{
    switch (Traits::producer_traits) {
    case ProducerTraits::multiple_producers:
        break;
    case ProducerTraits::single_producer:
        if (nproducers != 1) {
            std::cerr << "Using traits that support only one producer with more than one producer\n";
            exit(EXIT_FAILURE);
        }
        break;
    }
    switch (Traits::consumer_traits) {
    case ConsumerTraits::multiple_consumers:
        break;
    case ConsumerTraits::single_consumer:
        if (nconsumers != 1) {
            std::cerr << "Using traits that support only one consumer with more than one consumer\n";
            exit(EXIT_FAILURE);
        }
        break;
    }
}

struct TestDataNoExceptions
{
#if defined(USE_TBB_QUEUE)
    TestDataNoExceptions() noexcept
    : m_start(now())
    , m_copy_time()
    , m_payload(0)
    , m_thread_hash(0)
    {
        ++s_constructed;
    }
#endif

    // We pass this in in the constructor because we want to know the latency of enqueueing, not the latency of
    // construction/destruction..
    TestDataNoExceptions(int                v,
                         std::size_t        hash,
                         timing::duration_t construction_time,
                         timing::duration_t copy_time) noexcept
    : m_start(now())
    , m_copy_time(copy_time)
    , m_payload(v)
    , m_thread_hash(hash)
    {
        ++s_constructed;
        std::this_thread::sleep_for(construction_time);
    }

    TestDataNoExceptions(int v, std::size_t hash) noexcept
    : m_start(now())
    , m_copy_time(timing::duration_t::zero())
    , m_payload(v)
    , m_thread_hash(hash)
    {
        ++s_constructed;
    }

    TestDataNoExceptions(TestDataNoExceptions&& other) noexcept
    : m_start(other.m_start)
    , m_copy_time(other.m_copy_time)
    , m_payload(other.m_payload)
    , m_thread_hash(other.m_thread_hash)
    {
        ++s_constructed;
    }

    TestDataNoExceptions(const TestDataNoExceptions& other) noexcept
    : m_start(other.m_start)
    , m_copy_time(other.m_copy_time)
    , m_payload(other.m_payload)
    , m_thread_hash(other.m_thread_hash)
    {
        ++s_constructed;
        std::this_thread::sleep_for(m_copy_time);
    }

    ~TestDataNoExceptions() noexcept
    {
        --s_constructed;
    }

    TestDataNoExceptions& operator=(TestDataNoExceptions&& other) noexcept
    {
        // We implement this (instead of default) so that we can break.
        m_start       = other.m_start;
        m_copy_time   = other.m_copy_time;
        m_payload     = other.m_payload;
        m_thread_hash = other.m_thread_hash;
        return *this;
    }

    TestDataNoExceptions& operator=(const TestDataNoExceptions& other) noexcept
    {
        m_start       = other.m_start;
        m_copy_time   = other.m_copy_time;
        m_payload     = other.m_payload;
        m_thread_hash = other.m_thread_hash;
        std::this_thread::sleep_for(m_copy_time);
        return *this;
    }

    static timing::time_point_t now()
    {
        return timing::clock_t::now();
    }

    operator int() const noexcept
    {
        return m_payload;
    }

    timing::time_point_t   m_start;
    timing::duration_t     m_copy_time;
    int                    m_payload;
    std::size_t            m_thread_hash;
    static std::atomic_int s_constructed;
};

std::atomic_int TestDataNoExceptions::s_constructed(0);

struct TestDataExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::m_thread_hash;
    using TestDataNoExceptions::s_constructed;

    // We pass this in in the constructor because we want to know the latency of enqueueing, not the latency of
    // construction/destruction..
    TestDataExceptions(int                v,
                       std::size_t        hash,
                       timing::duration_t construction_time,
                       timing::duration_t copy_time,
                       bool               throw_in_construction,
                       bool               throw_in_copy_constructor,
                       bool               throw_in_copy_assignment,
                       bool               throw_in_move_constructor,
                       bool               throw_in_move_assignment)
    : TestDataNoExceptions(v, hash, construction_time, copy_time)
    , m_throw_in_copy_constructor(throw_in_copy_constructor)
    , m_throw_in_copy_assignment(throw_in_copy_assignment)
    , m_throw_in_move_constructor(throw_in_move_constructor)
    , m_throw_in_move_assignment(throw_in_move_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    TestDataExceptions(int v, std::size_t hash) noexcept
    : TestDataNoExceptions(v, hash)
    , m_throw_in_copy_constructor(false)
    , m_throw_in_copy_assignment(false)
    , m_throw_in_move_constructor(false)
    , m_throw_in_move_assignment(false)
    {
    }

    TestDataExceptions(TestDataExceptions&& other)
    : TestDataNoExceptions(std::move(other))
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_move_constructor) {
            throw TestException("Throw in move constructor");
        }
    }

    TestDataExceptions(const TestDataExceptions& other)
    : TestDataNoExceptions(other)
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_copy_constructor) {
            throw TestException("Throw in copy constructor");
        }
    }

    TestDataExceptions& operator=(TestDataExceptions&& other)
    {
        TestDataNoExceptions::operator=(std::move(other));
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_move_assignment) {
            throw TestException("Throw in move assignment");
        }
        return *this;
    }

    TestDataExceptions& operator=(const TestDataExceptions& other)
    {
        TestDataNoExceptions::operator=(other);
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_copy_assignment) {
            throw TestException("Throw in copy assignment");
        }
        return *this;
    }

    bool m_throw_in_copy_constructor;
    bool m_throw_in_copy_assignment;
    bool m_throw_in_move_constructor;
    bool m_throw_in_move_assignment;
};

struct TestDataCopyExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::m_thread_hash;
    using TestDataNoExceptions::s_constructed;

    // We pass this in in the constructor because we want to know the latency of enqueueing, not the latency of
    // construction/destruction..
    TestDataCopyExceptions(int                v,
                           std::size_t        hash,
                           timing::duration_t construction_time,
                           timing::duration_t copy_time,
                           bool               throw_in_construction,
                           bool               throw_in_copy_constructor,
                           bool               throw_in_copy_assignment)
    : TestDataNoExceptions(v, hash, construction_time, copy_time)
    , m_throw_in_copy_constructor(throw_in_copy_constructor)
    , m_throw_in_copy_assignment(throw_in_copy_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    TestDataCopyExceptions(int v, std::size_t hash) noexcept
    : TestDataNoExceptions(v, hash)
    , m_throw_in_copy_constructor(false)
    , m_throw_in_copy_assignment(false)
    {
    }

    TestDataCopyExceptions(TestDataCopyExceptions&& other) = default;

    TestDataCopyExceptions(const TestDataCopyExceptions& other)
    : TestDataNoExceptions(other)
    , m_throw_in_copy_constructor(other.m_throw_in_copy_constructor)
    , m_throw_in_copy_assignment(other.m_throw_in_copy_assignment)
    {
        if (m_throw_in_copy_constructor) {
            throw TestException("Throw in copy constructor");
        }
    }

    TestDataCopyExceptions& operator=(TestDataCopyExceptions&& other) noexcept = default;

    TestDataCopyExceptions& operator=(const TestDataCopyExceptions& other)
    {
        TestDataNoExceptions::operator=(other);
        m_throw_in_copy_constructor = other.m_throw_in_copy_constructor;
        m_throw_in_copy_assignment  = other.m_throw_in_copy_assignment;
        if (m_throw_in_copy_assignment) {
            throw TestException("Throw in copy assignment");
        }
        return *this;
    }

    bool m_throw_in_copy_constructor;
    bool m_throw_in_copy_assignment;
};

struct TestDataMoveExceptions : private TestDataNoExceptions
{
    using TestDataNoExceptions::now;
    using TestDataNoExceptions::operator int;
    using TestDataNoExceptions::m_copy_time;
    using TestDataNoExceptions::m_payload;
    using TestDataNoExceptions::m_start;
    using TestDataNoExceptions::m_thread_hash;
    using TestDataNoExceptions::s_constructed;

    // We pass this in in the constructor because we want to know the latency of enqueueing, not the latency of
    // construction/destruction..
    TestDataMoveExceptions(int                v,
                           std::size_t        hash,
                           timing::duration_t construction_time,
                           timing::duration_t copy_time,
                           bool               throw_in_construction,
                           bool               throw_in_move_constructor,
                           bool               throw_in_move_assignment)
    : TestDataNoExceptions(v, hash, construction_time, copy_time)
    , m_throw_in_move_constructor(throw_in_move_constructor)
    , m_throw_in_move_assignment(throw_in_move_assignment)
    {
        if (throw_in_construction) {
            throw TestException("Throw in constructor");
        }
    }

    TestDataMoveExceptions(int v, std::size_t hash) noexcept
    : TestDataNoExceptions(v, hash)
    , m_throw_in_move_constructor(false)
    , m_throw_in_move_assignment(false)
    {
    }

    TestDataMoveExceptions(TestDataMoveExceptions&& other)
    : TestDataNoExceptions(std::move(other))
    , m_throw_in_move_constructor(other.m_throw_in_move_constructor)
    , m_throw_in_move_assignment(other.m_throw_in_move_assignment)
    {
        if (m_throw_in_move_constructor) {
            throw TestException("Throw in move constructor");
        }
    }

    TestDataMoveExceptions(const TestDataMoveExceptions& other) noexcept = default;

    TestDataMoveExceptions& operator=(TestDataMoveExceptions&& other)
    {
        TestDataNoExceptions::operator=(std::move(other));
        m_throw_in_move_constructor = other.m_throw_in_move_constructor;
        m_throw_in_move_assignment  = other.m_throw_in_move_assignment;
        if (m_throw_in_move_assignment) {
            throw TestException("Throw in move assignment");
        }
        return *this;
    }

    TestDataMoveExceptions& operator=(const TestDataMoveExceptions& other) noexcept = default;

    bool m_throw_in_move_constructor;
    bool m_throw_in_move_assignment;
};

bool operator<(const TestDataNoExceptions& a, const TestDataNoExceptions& b)
{
    if (a.m_thread_hash < b.m_thread_hash) {
        return true;
    } else if (b.m_thread_hash < a.m_thread_hash) {
        return false;
    } else if (a.m_payload < b.m_payload) {
        return true;
    }
    return false;
}

bool operator<(const TestDataExceptions& a, const TestDataExceptions& b)
{
    if (a.m_thread_hash < b.m_thread_hash) {
        return true;
    } else if (b.m_thread_hash < a.m_thread_hash) {
        return false;
    } else if (a.m_payload < b.m_payload) {
        return true;
    }
    return false;
}

bool operator<(const TestDataCopyExceptions& a, const TestDataCopyExceptions& b)
{
    if (a.m_thread_hash < b.m_thread_hash) {
        return true;
    } else if (b.m_thread_hash < a.m_thread_hash) {
        return false;
    } else if (a.m_payload < b.m_payload) {
        return true;
    }
    return false;
}

bool operator<(const TestDataMoveExceptions& a, const TestDataMoveExceptions& b)
{
    if (a.m_thread_hash < b.m_thread_hash) {
        return true;
    } else if (b.m_thread_hash < a.m_thread_hash) {
        return false;
    } else if (a.m_payload < b.m_payload) {
        return true;
    }
    return false;
}

template <typename T, typename Traits>
using buffer_test_t = RingBufferImpl<T, k_log_queue_size_test, Traits>;

#if defined(PROFILE_RING_BUFFER)
template <typename T, typename Traits>
using buffer_profile_t = RingBufferImpl<T, k_log_queue_size_profile, Traits>;
#endif
#if defined(PROFILE_LOCKING_RINGBUFFER)
template <typename T, typename Traits>
using buffer_profile_t = LockingRingBuffer<T, k_log_queue_size_profile>;
#endif
#if defined(PROFILE_TBB_BOUNDED_QUEUE)
template <typename T, typename Traits>
using buffer_profile_t = tbb::concurrent_bounded_queue<T>;
#endif
#if defined(PROFILE_TBB_QUEUE)
template <typename T, typename Traits>
using buffer_profile_t = tbb::concurrent_queue<T>;
#endif

template <typename Duration>
class RandomTime
{
public:
    using rep = typename Duration::rep;

    RandomTime(double mean, double prob_of_extreme)
    : m_bernoulli(prob_of_extreme)
    , m_poisson(mean)
    {
    }

    template <typename RNG>
    Duration operator()(RNG& rng)
    {
        auto r = m_poisson(rng);
        if (m_bernoulli(rng)) {
            r *= 20;
        }

        return Duration(r);
    }

private:
    std::bernoulli_distribution    m_bernoulli;
    std::poisson_distribution<rep> m_poisson;
};

template <typename Duration, typename RNG>
Duration get_random_delay(RNG& rng)
{
#if 1
    return Duration::zero();
#else
    RandomTime<Duration> rt(1, 0.05f);
    return rt(rng);
#endif
}

template <typename IntType>
class WorkDistributor
{
public:
    struct Segment
    {
        Segment() noexcept
        : first(0)
        , last(0)
        , valid(false)
        {
        }

        Segment(IntType a, IntType b) noexcept
        : first(a)
        , last(b)
        , valid(a <= b)
        {
        }

        explicit operator bool() const noexcept
        {
            return valid;
        }

        IntType first;
        IntType last;
        bool    valid;
    };

    WorkDistributor(IntType nvalues, IntType nthreads) noexcept
    : m_last_start(0)
    , m_nvalues(nvalues)
    , m_nthreads(nthreads)
    {
    }

    // This is a mutable function. This class acts as a generator.
    Segment operator()() noexcept
    {
        if (m_nthreads > 0) {
            const IntType v = m_nvalues / m_nthreads;
            m_nvalues -= v;
            --m_nthreads;
            const IntType start = m_last_start;
            const IntType end   = start + v;
            m_last_start        = end;
            return Segment{start, end};
        }

        return Segment{};
    }

private:
    IntType m_last_start;
    IntType m_nvalues;
    IntType m_nthreads;
};

template <typename IntType>
WorkDistributor<IntType> make_work_distributor(IntType nvalues, IntType nthreads) noexcept
{
    return WorkDistributor<IntType>{nvalues, nthreads};
}

template <typename DataType>
class Emplacer
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, std::size_t thread_hash, RNG& rng);
};

template <>
class Emplacer<TestDataNoExceptions>
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, std::size_t thread_hash, RNG& rng) const
    {
        using duration_type = std::chrono::milliseconds;

        buffer.emplace(value, thread_hash, get_random_delay<duration_type>(rng), get_random_delay<duration_type>(rng));
    }
};

template <>
class Emplacer<TestDataExceptions>
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, std::size_t thread_hash, RNG& rng)
    {
        using duration_type = std::chrono::milliseconds;

        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_constructor));
        const bool throw_copy_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_assignment));
        const bool throw_copy_assignment = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_constructor));
        const bool throw_move_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_assignment));
        const bool throw_move_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       thread_hash,
                       get_random_delay<duration_type>(rng),
                       get_random_delay<duration_type>(rng),
                       throw_construct,
                       throw_copy_constructor,
                       throw_copy_assignment,
                       throw_move_constructor,
                       throw_move_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <>
class Emplacer<TestDataCopyExceptions>
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, std::size_t thread_hash, RNG& rng)
    {
        using duration_type = std::chrono::milliseconds;

        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_constructor));
        const bool throw_copy_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_copy_assignment));
        const bool throw_copy_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       thread_hash,
                       get_random_delay<duration_type>(rng),
                       get_random_delay<duration_type>(rng),
                       throw_construct,
                       throw_copy_constructor,
                       throw_copy_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <>
class Emplacer<TestDataMoveExceptions>
{
public:
    template <typename BufferType, typename RNG>
    void do_emplace(BufferType& buffer, int value, std::size_t thread_hash, RNG& rng)
    {
        using duration_type = std::chrono::milliseconds;

        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_construct));
        const bool throw_construct = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_constructor));
        const bool throw_move_constructor = m_bernoulli_distribution(rng);
        m_bernoulli_distribution.param(std::bernoulli_distribution::param_type(p_throw_move_assignment));
        const bool throw_move_assignment = m_bernoulli_distribution(rng);

        buffer.emplace(value,
                       thread_hash,
                       get_random_delay<duration_type>(rng),
                       get_random_delay<duration_type>(rng),
                       throw_construct,
                       throw_move_constructor,
                       throw_move_assignment);
    }

private:
    std::bernoulli_distribution m_bernoulli_distribution;
};

template <typename T, typename Traits>
class ProducerTest
{
    using RNG = std::mt19937;

public:
    ProducerTest(int first, int last, buffer_test_t<T, Traits>& buffer) noexcept
    : m_first(first)
    , m_last(last)
    , m_buffer(std::addressof(buffer))
    {
        std::seed_seq seed{first, last};
        m_rng.seed(seed);
    }

    void operator()()
    {
        using hash                    = std::hash<std::thread::id>;
        const std::size_t thread_hash = hash{}(std::this_thread::get_id());
        Emplacer<T>       emplacer;

        for (int i = m_first; i < m_last;) {
            try {
                emplacer.do_emplace(*m_buffer, i, thread_hash, m_rng);
                ++i;
            } catch (TestException& e) {
                std::cerr << "Testing exception: " << e.what() << '\n';
            } catch (std::exception& e) {
                std::cerr << "Unexpected exception: " << e.what() << '\n';
                throw;
            } catch (...) {
                std::cerr << "Unknown exception\n";
                throw;
            }
        }
    }

private:
    int                       m_first;
    int                       m_last;
    RNG                       m_rng;
    buffer_test_t<T, Traits>* m_buffer;
};

template <typename T, typename Traits>
class ConsumerTest
{
public:
    ConsumerTest(int nv, buffer_test_t<T, Traits>& buffer)
    : m_nvalues(nv)
    , m_buffer(std::addressof(buffer))
    {
    }

    void operator()()
    {
        for (int i = 0; i < m_nvalues; ++i) {
            try {
                const auto v = m_buffer->pop();
                g_stats.push(v.now() - v.m_start);
                const int as_int = v;
                m_values.push_back(as_int);
                // print_safe("consuming ", as_int);
            } catch (TestException& e) {
                std::cerr << "Testing Exception: " << e.what() << '\n';
            } catch (std::exception& e) {
                std::cerr << "Unknown exception: " << e.what() << '\n';
                throw;
            } catch (...) {
                std::cerr << "Unknown exception\n";
                throw;
            }
        }
    }

    std::vector<int>          m_values;
    int                       m_nvalues;
    buffer_test_t<T, Traits>* m_buffer;
};

template <typename T, typename Traits>
struct ProducerProfile
{
    using BufferType = buffer_profile_t<T, Traits>;

    ProducerProfile(int first, int last, BufferType& buffer) noexcept
    : m_first(first)
    , m_last(last)
    , m_buffer(std::addressof(buffer))
    {
    }

    void operator()()
    {
        using hash                    = std::hash<std::thread::id>;
        constexpr int     k_max_value = 4096;
        const std::size_t thread_hash = hash{}(std::this_thread::get_id());

        for (int i = m_first; i < m_last; ++i) {
            const int value = i % k_max_value;
            m_stopwatch.start();
            m_buffer->emplace(value, thread_hash);
            m_stopwatch.stop();
        }
    }

    auto get_duration() const
    {
        return m_stopwatch.duration();
    }

    int         m_first;
    int         m_last;
    StopWatch   m_stopwatch;
    BufferType* m_buffer;
};

template <typename T, typename Traits>
struct ConsumerProfile
{
    using BufferType = buffer_profile_t<T, Traits>;

    ConsumerProfile(int nv, BufferType& buffer)
    : m_nvalues(nv)
    , m_buffer(std::addressof(buffer))
    {
    }

    ConsumerProfile(const ConsumerProfile&)            = delete;
    ConsumerProfile(ConsumerProfile&&)                 = default;
    ConsumerProfile& operator=(const ConsumerProfile&) = delete;
    ConsumerProfile& operator=(ConsumerProfile&&)      = default;

    void operator()()
    {
        for (int i = 0; i < m_nvalues; ++i) {
            m_stopwatch.start();
#if defined(USE_TBB_QUEUE)
            T v;
            while (!m_buffer->try_pop(v))
                ;
#else
            const auto v = m_buffer->pop();
#endif
            m_stopwatch.stop();
            ++m_values[v];
        }
    }

    auto get_duration() const
    {
        return m_stopwatch.duration();
    }

    std::map<T, int> m_values;
    int              m_nvalues;
    StopWatch        m_stopwatch;
    BufferType*      m_buffer;
};

std::atomic_uint loop_iterations(0);

template <typename T, typename Traits>
void do_test(int nproducers, int nconsumers)
{
    g_stats.reset();

    verify_preconditions<Traits>(nproducers, nconsumers);
    if (T::s_constructed != 0) {
        T::s_constructed = 0;
    }

    {
        buffer_test_t<T, Traits> buffer;

        std::cout << "------------------------\n";
        std::cout << "Testing with ";
        print_traits_info<Traits>(std::cout);
        std::cout << "Using resolved lock_policy: ";
        print_lock_traits(std::cout, buffer_test_t<T, Traits>::get_lock_traits());
        std::cout << "With " << nproducers << " producer(s) and " << nconsumers << " consumer(s)" << std::endl;

        std::vector<std::thread> producer_threads;
        producer_threads.reserve(nproducers);

        std::vector<std::thread> consumer_threads;
        consumer_threads.reserve(nconsumers);

        StopWatch watch;
        watch.start();

        std::vector<ConsumerTest<T, Traits>> consumers;
        auto                                 consumer_distributor = make_work_distributor(k_nvalues, nconsumers);
        while (auto segment = consumer_distributor()) {
            consumers.emplace_back(segment.last - segment.first, buffer);
        }

        auto producer_distributor = make_work_distributor(k_nvalues, nproducers);
        while (auto segment = producer_distributor()) {
            producer_threads.emplace_back(ProducerTest<T, Traits>{segment.first, segment.last, buffer});
        }

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads.emplace_back(std::ref(consumers[i]));
        }
        for (int i = 0; i < nproducers; ++i) {
            producer_threads[i].join();
        }
        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads[i].join();
        }

        watch.stop();
        std::cout << watch.count() << " seconds\n";
        std::cout << g_stats.mean() << " mean seconds\n";
        std::cout << loop_iterations << " iterations\n";

        std::vector<int> results;
        for (int i = 0; i < nconsumers; ++i) {
            results.insert(results.end(), consumers[i].m_values.cbegin(), consumers[i].m_values.cend());
        }
        std::ranges::sort(results);
        for (int i = 0; i < k_nvalues; ++i) {
            if (i != results[i]) {
                std::cerr << "Can't find " << i << '\n';
                break;
            }
        }
    }
    if (T::s_constructed != 0) {
        std::cerr << "End: unexpected number constructed: " << T::s_constructed << '\n';
    }
}

void test(int nproducers, int nconsumers)
{
    const int total_threads = nproducers + nconsumers;
    do_test<TestDataNoExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataNoExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataNoExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataNoExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataCopyExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataCopyExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataCopyExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataCopyExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);

    do_test<TestDataMoveExceptions, DefaultRingBufferTraits>(nproducers, nconsumers);
    do_test<TestDataMoveExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
    do_test<TestDataMoveExceptions, SingleProducerRingBufferTraits>(1, total_threads);
    do_test<TestDataMoveExceptions, SingleConsumerRingBufferTraits>(total_threads, 1);
}

void test()
{
    const int hw_threads = std::thread::hardware_concurrency();
    const int nproducers = hw_threads / 2;
    const int nconsumers = hw_threads - nproducers;
    test(nproducers, nconsumers);
}

template <typename Traits>
void test_batch(int nconsumers, int batch_size)
{
    verify_preconditions<Traits>(1, nconsumers);
    std::cout << "Testing batch with ";
    print_traits_info<Traits>(std::cout);
    std::cout << "With " << nconsumers << " consumer(s)" << std::endl;

    static_assert(Traits::producer_traits == ProducerTraits::single_producer);

    try {
        buffer_test_t<TestDataNoExceptions, Traits> buffer;

        std::vector<std::thread> consumer_threads;
        consumer_threads.reserve(nconsumers);

        std::vector<ConsumerTest<TestDataNoExceptions, Traits>> consumers;
        auto consumer_distributor = make_work_distributor(k_nvalues, nconsumers);
        while (auto segment = consumer_distributor()) {
            consumers.emplace_back(segment.last - segment.first, buffer);
        }

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads.emplace_back(std::ref(consumers[i]));
        }

        using hash                    = std::hash<std::thread::id>;
        const std::size_t thread_hash = hash{}(std::this_thread::get_id());

        std::vector<TestDataNoExceptions> values;
        values.reserve(k_nvalues);
        for (int i = 0; i < k_nvalues; ++i) {
            values.emplace_back(i, thread_hash);
        }

        /////
        batch_size = std::min(k_nvalues, std::min(static_cast<int>(buffer.capacity()), batch_size));
        auto first = values.cbegin();
        auto next  = first + batch_size;
        while (first != values.cend()) {
            buffer.push_batch(first, next);
            const int nleft = std::distance(next, values.cend());
            const int d     = std::min(nleft, batch_size);
            first           = next;
            std::advance(next, d);
        }
        /////

        for (int i = 0; i < nconsumers; ++i) {
            consumer_threads[i].join();
        }

        std::vector<int> results;
        for (int i = 0; i < nconsumers; ++i) {
            results.insert(results.end(), consumers[i].m_values.cbegin(), consumers[i].m_values.cend());
        }
        std::ranges::sort(results);
        for (int i = 0; i < k_nvalues; ++i) {
            if (i != results[i]) {
                std::cout << "Can't find " << i << '\n';
                break;
            }
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed.\n";
    } catch (...) {
        std::cerr << "Unknown error\n";
    }
    std::cout << "Done with test" << std::endl;
}

void test_batch(int nconsumers, int batch_size)
{
    nconsumers = std::max(1, nconsumers);
    test_batch<SingleProducerRingBufferTraits>(nconsumers, batch_size);
    test_batch<SingleProducerSingleConsumerRingBufferTraits>(1, batch_size);
}

void test_batch(int batch_size)
{
    const int hw_threads = std::thread::hardware_concurrency();
    test_batch(hw_threads, batch_size);
}

void stress_test()
{
    constexpr int rounds = 32;

    const int hw_threads = std::thread::hardware_concurrency();
    for (int i = 0; i < rounds; ++i) {
        test_batch<SingleProducerSingleConsumerRingBufferTraits>(1, 32);
        test_batch<SingleProducerSingleConsumerRingBufferTraits>(1, k_nvalues);
        for (int j = 1; j < hw_threads; ++j) {
            test_batch<SingleProducerRingBufferTraits>(j, 32);
            test_batch<SingleProducerRingBufferTraits>(j, k_nvalues);
        }
        do_test<TestDataNoExceptions, DefaultRingBufferTraits>(1, 1);
        do_test<TestDataNoExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
        do_test<TestDataNoExceptions, SingleProducerRingBufferTraits>(1, 1);
        do_test<TestDataNoExceptions, SingleConsumerRingBufferTraits>(1, 1);

        do_test<TestDataExceptions, DefaultRingBufferTraits>(1, 1);
        do_test<TestDataExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
        do_test<TestDataExceptions, SingleProducerRingBufferTraits>(1, 1);
        do_test<TestDataExceptions, SingleConsumerRingBufferTraits>(1, 1);

        do_test<TestDataCopyExceptions, DefaultRingBufferTraits>(1, 1);
        do_test<TestDataCopyExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
        do_test<TestDataCopyExceptions, SingleProducerRingBufferTraits>(1, 1);
        do_test<TestDataCopyExceptions, SingleConsumerRingBufferTraits>(1, 1);

        do_test<TestDataMoveExceptions, DefaultRingBufferTraits>(1, 1);
        do_test<TestDataMoveExceptions, SingleProducerSingleConsumerRingBufferTraits>(1, 1);
        do_test<TestDataMoveExceptions, SingleProducerRingBufferTraits>(1, 1);
        do_test<TestDataMoveExceptions, SingleConsumerRingBufferTraits>(1, 1);
        for (int j = 1; j < hw_threads; ++j) {
            const int k = hw_threads - j;
            do_test<TestDataNoExceptions, DefaultRingBufferTraits>(j, k);
            do_test<TestDataNoExceptions, SingleProducerRingBufferTraits>(1, j);
            do_test<TestDataNoExceptions, SingleConsumerRingBufferTraits>(j, 1);

            do_test<TestDataExceptions, DefaultRingBufferTraits>(j, k);
            do_test<TestDataExceptions, SingleProducerRingBufferTraits>(1, j);
            do_test<TestDataExceptions, SingleConsumerRingBufferTraits>(j, 1);

            do_test<TestDataCopyExceptions, DefaultRingBufferTraits>(j, k);
            do_test<TestDataCopyExceptions, SingleProducerRingBufferTraits>(1, j);
            do_test<TestDataCopyExceptions, SingleConsumerRingBufferTraits>(j, 1);

            do_test<TestDataMoveExceptions, DefaultRingBufferTraits>(j, k);
            do_test<TestDataMoveExceptions, SingleProducerRingBufferTraits>(1, j);
            do_test<TestDataMoveExceptions, SingleConsumerRingBufferTraits>(j, 1);
        }
    }
}

struct ProfileData
{
    static ProfileData max()
    {
        ProfileData pd;
        pd.producer_duration = StopWatch::duration_t::max();
        pd.consumer_duration = StopWatch::duration_t::max();
        return pd;
    }

    StopWatch::duration_t producer_duration = StopWatch::duration_t::zero();
    StopWatch::duration_t consumer_duration = StopWatch::duration_t::zero();
};

template <typename T>
ProfileData do_profile(int nproducers, int nconsumers)
{
#if defined(PROFILE_LOCK_FREE)
    using Traits = LockFreeTraits;
#else
    using Traits = DefaultRingBufferTraits;
#endif
    buffer_profile_t<T, Traits> buffer;

#if defined(EXPLICIT_TBB_BOUNDED_QUEUE_SIZE)
    buffer.set_capacity(1 << k_log_queue_size_profile);
#endif

    std::vector<ProducerProfile<T, Traits>> producers;
    std::vector<ConsumerProfile<T, Traits>> consumers;

    auto consumer_distributor = make_work_distributor(k_nvalues, nconsumers);
    while (auto segment = consumer_distributor()) {
        consumers.emplace_back(segment.last - segment.first, buffer);
    }

    auto producer_distributor = make_work_distributor(k_nvalues, nproducers);
    while (auto segment = producer_distributor()) {
        producers.emplace_back(segment.first, segment.last, buffer);
    }

    std::vector<std::thread> producer_threads;
    producer_threads.reserve(nproducers);

    std::vector<std::thread> consumer_threads;
    consumer_threads.reserve(nconsumers);

    for (int i = 0; i < nconsumers; ++i) {
        consumer_threads.emplace_back(std::ref(consumers[i]));
    }
    for (int i = 0; i < nproducers; ++i) {
        producer_threads.emplace_back(std::ref(producers[i]));
    }
    for (int i = 0; i < nproducers; ++i) {
        producer_threads[i].join();
    }
    for (int i = 0; i < nconsumers; ++i) {
        consumer_threads[i].join();
    }

    int                sum = 0;
    std::map<int, int> results;
    std::map<T, int>   results_with_thread;
    for (int i = 0; i < nconsumers; ++i) {
        for (const auto& v : consumers[i].m_values) {
            results_with_thread[v.first] += v.second;
            results[v.first.m_payload] += v.second;
            sum += v.second;
        }
    }
    if (sum != k_nvalues) {
        std::cerr << "Error detected\n";
    }
    // for (const auto& v : results_with_thread) {
    // std::cout << v.first.m_thread_hash << " <--> " << v.first.m_payload << " -> " << v.second << '\n';
    //}
    // for (const auto& v : results) {
    // std::cout << v.first << " -> " << v.second << '\n';
    //}

    ProfileData timing_results;
    for (const auto& p : producers) {
        timing_results.producer_duration += p.get_duration();
    }
    for (const auto& c : consumers) {
        timing_results.consumer_duration += c.get_duration();
    }
    return timing_results;
}

template <typename T>
void profile(int nproducers, int nconsumers)
{
    if (k_verbose) {
        std::cout << "Profiling with " << nproducers << " producers and " << nconsumers << " consumers" << std::endl;
    } else {
        std::cout << "# " << nproducers << ' ' << nconsumers << ' ';
    }

    for (int i = 0; i < k_profile_count; ++i) {
        const auto results = do_profile<T>(nproducers, nconsumers);
        std::cout << results.producer_duration.count() << '\t' << results.consumer_duration.count() << '\n';
    }
#if 0
    auto min_values = ProfileData::max();
    for (int i = 0; i < k_profile_count; ++i) {
        const auto results = do_profile(nproducers, nconsumers);
        min_values.producer_duration = std::min(min_values.producer_duration, results.producer_duration);
        min_values.consumer_duration = std::min(min_values.consumer_duration, results.consumer_duration);
        if (k_verbose) {
            std::cout << "Push time: " << results.producer_duration.count() << '\n';
            std::cout << "Pop time: " << results.consumer_duration.count() << '\n';
        }
    }
    std::cout << min_values.producer_duration.count() << ' ' << min_values.consumer_duration.count() << '\n';
#endif
}

template <typename T>
void profile()
{
    const int hw_threads = std::thread::hardware_concurrency();
    const int nproducers = hw_threads / 2;
    const int nconsumers = hw_threads - nproducers;

    profile<T>(nproducers, nconsumers);
}

template <typename T>
void profile_all_the_things()
{
    const int hw_threads   = std::thread::hardware_concurrency();
    const int half_threads = hw_threads / 2;

    std::ofstream outs_producers("producers.dat");
    std::ofstream outs_consumers("consumers.dat");

    if (!outs_producers || !outs_consumers) {
        std::cerr << "Unable to open all output files\n";
        return;
    }

    outs_producers << "# " << get_stopwatch_units() << '\n';
    outs_consumers << "# " << get_stopwatch_units() << '\n';

    outs_producers << '#';
    outs_consumers << '#';
    for (int i = 1; i <= half_threads; ++i) {
        outs_producers << '\t' << i;
        outs_consumers << '\t' << i;
    }
    outs_producers.put('\n');
    outs_consumers.put('\n');

    constexpr auto precision = std::numeric_limits<StopWatch::representation_t>::digits10 + 1;

    for (int p = 0; p < k_profile_count; ++p) {
        std::cout << "Profiling " << (p + 1) << " of " << k_profile_count << '\n';
        for (int i = 1; i < half_threads; ++i) {
            const auto results = do_profile<T>(i, i);
            if (i > 1) {
                outs_producers << '\t';
                outs_consumers << '\t';
            }
            outs_producers << std::fixed << std::setprecision(precision) << results.producer_duration.count();
            outs_consumers << std::fixed << std::setprecision(precision) << results.consumer_duration.count();
        }
        outs_producers.put('\n');
        outs_consumers.put('\n');
    }
}

void print_usage(const char* exe_name)
{
    std::cout << "Usage: " << exe_name << "[#producers #consumers]\n";
}

int main(int argc, const char* const argv[])
{
    static_assert(RingBuffer<TestDataNoExceptions, 1>::get_lock_traits() == LockTraits::spin_lock);
    static_assert(RingBuffer<TestDataExceptions, 1>::get_lock_traits() == LockTraits::lock_free);

    if (argc == 1) {
        test();
        test_batch(32);
        test_batch(k_nvalues);
        //stress_test();
        // profile<TestDataNoExceptions>();
    } else if (argc == 2) {
        const std::string arg(argv[1]);
        if (arg == "--unit") {
            std::cout << get_stopwatch_units() << '\n';
            return EXIT_SUCCESS;
        } else if (arg == "--all") {
            std::cout << "*********** Profile No Exceptions ***********\n";
            profile_all_the_things<TestDataNoExceptions>();
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } else if (argc == 3) {
        const int nproducers = std::atoi(argv[1]);
        const int nconsumers = std::atoi(argv[2]);
        if (nproducers == 0 || nconsumers == 0) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::cout << "*********** Test ***********\n";
        test(nproducers, nconsumers);
        std::cout << "*********** Test Batch ***********\n";
        test_batch(nconsumers);
        std::cout << "*********** Profile No Exceptions ***********\n";
        profile<TestDataNoExceptions>(nproducers, nconsumers);
    } else {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // render(501, 463);
}
