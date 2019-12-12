#pragma once
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename Dim>
using basic_host_tensor =
    basic_tensor<R, basic_shape<r, Dim>, host_memory, owner>;

template <typename R, rank_t r, typename Dim>
using basic_host_tensor_ref =
    basic_tensor<R, basic_shape<r, Dim>, host_memory, readwrite>;

template <typename R, rank_t r, typename Dim>
using basic_host_tensor_view =
    basic_tensor<R, basic_shape<r, Dim>, host_memory, readonly>;
}  // namespace internal
}  // namespace ttl
