#pragma once
#include <memory>

#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R> class basic_allocator<R, host_memory>
{
  public:
    R *operator()(size_t count) { return new R[count]; }
};

template <typename R> class basic_deallocator<R, host_memory>
{
  public:
    void operator()(R *data) { delete[] data; }
};

template <typename R, typename D>
using own_ptr = std::unique_ptr<R[], basic_deallocator<R, D>>;

template <typename R> class ref_ptr
{
    R *ptr_;

  public:
    using ptr_type = R *;
    using ref_type = R &;

    ref_ptr(ptr_type ptr) : ptr_(ptr) {}

    R *get() const { return ptr_; }
};

template <typename R> class view_ptr
{
    const R *ptr_;

  public:
    using ptr_type = const R *;
    using ref_type = const R &;

    view_ptr(ptr_type ptr) : ptr_(ptr) {}

    const R *get() const { return ptr_; }
};
}  // namespace internal
}  // namespace ttl
