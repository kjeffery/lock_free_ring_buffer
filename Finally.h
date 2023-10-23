//
// Created by Keith on 4/6/2021.
//

#pragma once

///@author Keith Jeffery

#include <utility>

#if !defined(__CUDA_ARCH__)
template <typename F>
class Finally
{
public:
    explicit Finally(F f)
    : m_f(std::move(f))
    {
    }

    ~Finally() noexcept
    {
        m_f();
    }

private:
    F m_f;
};

template <typename F>
Finally<F> finally(F f)
{
    return Finally<F>(std::move(f));
}
#else
template <typename F>
class Finally
{
public:
    __host__ __device__ explicit Finally(F f)
    : m_f(std::move(f))
    {
    }

    __host__ __device__ ~Finally() noexcept
    {
        m_f();
    }

private:
    F m_f;
};

template <typename F>
__host__ __device__ Finally<F> finally(F f)
{
    return Finally<F>(std::move(f));
}
#endif

