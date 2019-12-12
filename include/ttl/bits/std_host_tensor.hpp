#pragma once
#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename Dim>
using basic_host_tensor =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, host_memory,
                           internal::owner>;

template <typename R, rank_t r, typename Dim>
using basic_host_tensor_ref =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, host_memory,
                           internal::readwrite>;

template <typename R, rank_t r, typename Dim>
using basic_host_tensor_view =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, host_memory,
                           internal::readonly>;
}  // namespace internal
}  // namespace ttl
