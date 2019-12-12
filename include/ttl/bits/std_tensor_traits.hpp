#pragma once
#include <memory>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
struct owner;
struct readwrite;
struct readonly;

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

template <typename R> struct basic_tensor_traits<R, owner> {
    using ptr_type = R *;
    using ref_type = R &;

    using D = ref_ptr<R>;  // FIXME: use own_ptr
    using Access = readwrite;
};

template <typename R> struct basic_tensor_traits<R, readwrite> {
    using ptr_type = R *;
    using ref_type = R &;

    using D = ref_ptr<R>;
    using Access = readwrite;
};

template <typename R> struct basic_tensor_traits<R, readonly> {
    using ptr_type = const R *;
    using ref_type = const R &;

    using D = view_ptr<R>;
    using Access = readonly;
};
}  // namespace internal
}  // namespace ttl
