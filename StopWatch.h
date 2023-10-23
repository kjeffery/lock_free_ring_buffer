//
// Created by Keith on 12/7/2020.
//

#pragma once

#include <chrono>

class StopWatch
{
public:
    using representation_t = long double;
    using clock_t          = std::chrono::steady_clock;
    using time_point_t     = std::chrono::time_point<clock_t>;
    using duration_t       = std::chrono::duration<representation_t>;
    using period           = clock_t::period;

    StopWatch();
    void start();
    void stop();
    duration_t duration() const;
    representation_t count() const;
    void restart();

private:
    time_point_t mStart;
    duration_t   mDuration;
};

inline StopWatch::StopWatch()
: mDuration(duration_t::zero())
{
}

inline void StopWatch::start()
{
    mStart = clock_t::now();
}

inline void StopWatch::stop()
{
    time_point_t tp = clock_t::now();
    mDuration += tp - mStart;
}

inline StopWatch::duration_t StopWatch::duration() const
{
    return mDuration;
}

inline StopWatch::representation_t StopWatch::count() const
{
    return mDuration.count();
}

inline void StopWatch::restart()
{
    mDuration = duration_t::zero();
}

