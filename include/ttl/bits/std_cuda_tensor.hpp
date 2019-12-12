#pragma once
#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, cuda_memory,
                           internal::owner>;

template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor_ref =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, cuda_memory,
                           internal::readwrite>;

template <typename R, rank_t r, typename Dim>
using basic_cuda_tensor_view =
    internal::basic_tensor<R, internal::basic_shape<r, Dim>, cuda_memory,
                           internal::readonly>;
}  // namespace internal
}  // namespace ttl
