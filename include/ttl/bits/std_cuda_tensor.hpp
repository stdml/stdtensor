#pragma once
#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor =
    basic_tensor<R, basic_shape<r, Dim>, cuda_memory, owner>;

template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor_ref =
    basic_tensor<R, basic_shape<r, Dim>, cuda_memory, readwrite>;

template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor_view =
    basic_tensor<R, basic_shape<r, Dim>, cuda_memory, readonly>;
}  // namespace internal
}  // namespace ttl
