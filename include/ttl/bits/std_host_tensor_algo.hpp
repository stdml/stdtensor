#pragma once
#include <algorithm>
#include <functional>

#include <ttl/bits/std_host_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename Dim>
Dim argmax(const basic_host_tensor_view<R, 1, Dim> &t)
{
    return std::max_element(t.data(), t.data_end()) - t.data();
}

template <typename R, typename R1, rank_t r, typename Dim>
void cast(const basic_host_tensor_view<R, r, Dim> &x,
          const basic_host_tensor_ref<R1, r, Dim> &y)
{
    std::transform(x.data(), x.data_end(), y.data(),
                   [](const R &e) -> R1 { return static_cast<R1>(e); });
}

template <typename R, rank_t r, typename Dim>
void fill(const basic_host_tensor_ref<R, r, Dim> &t, const R &x)
{
    std::fill(t.data(), t.data_end(), x);
}

template <typename R, rank_t r, typename Dim>
Dim hamming_distance(const basic_host_tensor_view<R, r, Dim> &x,
                     const basic_host_tensor_view<R, r, Dim> &y)
{
    return std::inner_product(x.data(), x.data_end(), y.data(),
                              static_cast<Dim>(0), std::plus<Dim>(),
                              std::not_equal_to<R>());
}

template <typename R, rank_t r, typename Dim>
R chebyshev_distenace(const basic_host_tensor_view<R, r, Dim> &x,
                      const basic_host_tensor_view<R, r, Dim> &y)
{
    return std::inner_product(
        x.data(), x.data_end(), y.data(), static_cast<R>(0),
        [](R a, R d) { return std::max<R>(a, d); },
        [](R x, R y) {
            // FIXME: make sure it is commutative for floats
            return x > y ? x - y : y - x;
        });
}

template <typename R, rank_t r, typename Dim>
R max(const basic_host_tensor_view<R, r, Dim> &t)
{
    return *std::max_element(t.data(), t.data_end());
}

template <typename R, rank_t r, typename Dim>
R min(const basic_host_tensor_view<R, r, Dim> &t)
{
    return *std::min_element(t.data(), t.data_end());
}

template <typename R, rank_t r, typename Dim>
R sum(const basic_host_tensor_view<R, r, Dim> &t)
{
    return std::accumulate(t.data(), t.data_end(), static_cast<R>(0));
}

template <typename R, rank_t r, typename Dim>
R mean(const basic_host_tensor_view<R, r, Dim> &t)
{
    return sum(t) / static_cast<R>(t.shape().size());
}
}  // namespace internal
}  // namespace ttl
