#pragma once
#include <cstring>

#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{
template <>
class basic_copier<host_memory, host_memory>
{
  public:
    void operator()(void *dst, const void *src, size_t size)
    {
        std::memcpy(dst, src, size);
    }
};

namespace experimental
{
template <typename D1, typename D2>
void memcpy(void *dst, const void *src, size_t size)
{
    basic_copier<D1, D2>()(dst, src, size);
}

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
