#pragma once
#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
namespace experimental
{
template <typename R, typename S>
void copy(const basic_tensor<R, S, host_memory, readwrite> &dst,
          const basic_tensor<R, S, cuda_memory, readonly> &src)
{
    using copier = internal::cuda_copier;
    copier::copy<copier::d2h>(dst.data(), src.data(), src.data_size());
}

template <typename R, typename S>
void copy(const basic_tensor<R, S, cuda_memory, readwrite> &dst,
          const basic_tensor<R, S, host_memory, readonly> &src)
{
    using copier = internal::cuda_copier;
    copier::copy<copier::h2d>(dst.data(), src.data(), src.data_size());
}
}  // namespace experimental
}  // namespace internal
}  // namespace ttl
