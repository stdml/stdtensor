#pragma once
#include <stdml/tensor_config.hpp>
#include <ttl/bits/flat_shape.hpp>

namespace stdml
{
template <typename T, typename C>
std::vector<T> cast_to(const C &xs)
{
    std::vector<T> ys(xs.size());
    std::copy(xs.begin(), xs.end(), ys.begin());
    return ys;
}

template <typename T, size_t r, typename C>
std::array<T, r> cast_to(const C &xs)
{
    if (xs.size() != r) { throw std::invalid_argument(__func__); }
    std::array<T, r> ys;
    std::copy(xs.begin(), xs.end(), ys.begin());
    return ys;
}

class Shape
{
    using S = flat_shape;

    const S s_;

  public:
    Shape() {}

    Shape(const std::list<long> &dims) : s_(cast_to<int64_t>(dims)) {}

    template <typename D, size_t r>
    Shape(const std::array<D, r> &dims) : s_(cast_to<int64_t>(dims))
    {
    }

    Shape(S s) : s_(std::move(s)) {}

    template <ttl::rank_t r>
    Shape(const ttl::shape<r> &s) : s_(s)
    {
    }

    // operator S() const { return s_; }

    const S &get() const { return s_; }

    size_t rank() const { return s_.rank(); }

    size_t size() const { return s_.size(); }

    template <ttl::rank_t r, typename Dim = dim_t>
    ttl::internal::basic_shape<r, Dim> ranked() const
    {
        std::vector<Dim> dims = cast_to<Dim>(s_.dims());
        ttl::internal::basic_flat_shape<Dim> s(std::move(dims));
        return s.template ranked<r>();
    }
};
}  // namespace stdml
