#pragma once
#include <cstddef>

#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R>
class basic_allocator<R, host_memory>
{
  public:
    R *operator()(size_t count) { return new R[count]; }
};

template <typename R>
class basic_deallocator<R, host_memory>
{
  public:
    void operator()(R *data) { delete[] data; }
};
}  // namespace internal
}  // namespace ttl
