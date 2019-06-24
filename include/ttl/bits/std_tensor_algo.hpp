#pragma once
#include <algorithm>
#include <functional>

#include <ttl/bits/std_shape.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D = typename S::dimension_type>
D argmax(const basic_tensor_view<R, 1, S> &t)
{
    return std::max_element(t.data(), t.data_end()) - t.data();
}

template <typename R, typename R1, rank_t r, typename S>
void cast(const basic_tensor_view<R, r, S> &x,
          const basic_tensor_ref<R1, r, S> &y)
{
    std::transform(x.data(), x.data_end(), y.data(),
                   [](const R &e) -> R1 { return static_cast<R1>(e); });
}

template <typename R, rank_t r, typename S>
void fill(const basic_tensor_ref<R, r, S> &t, const R &x)
{
    std::fill(t.data(), t.data_end(), x);
}

template <typename R, rank_t r, typename S,
          typename D = typename S::dimension_type>
D hamming_distance(const basic_tensor_view<R, r, S> &x,
                   const basic_tensor_view<R, r, S> &y)
{
    return std::inner_product(x.data(), x.data_end(), y.data(),
                              static_cast<D>(0), std::plus<D>(),
                              std::not_equal_to<R>());
}

template <typename R, rank_t r, typename S>
R max(const basic_tensor_view<R, r, S> &t)
{
    return *std::max_element(t.data(), t.data_end());
}

template <typename R, rank_t r, typename S>
R min(const basic_tensor_view<R, r, S> &t)
{
    return *std::min_element(t.data(), t.data_end());
}

template <typename R, rank_t r, typename S>
R sum(const basic_tensor_view<R, r, S> &t)
{
    return std::accumulate(t.data(), t.data_end(), static_cast<R>(0));
}

template <typename R, rank_t r, typename S>
R mean(const basic_tensor_view<R, r, S> &t)
{
    return sum(t) / static_cast<R>(t.shape().size());
}

}  // namespace internal
}  // namespace ttl
