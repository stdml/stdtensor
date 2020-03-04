#pragma once
#include <memory>
#include <ttl/bits/std_access_traits.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename D>
using own_ptr = std::unique_ptr<R[], basic_deallocator<R, D>>;

template <typename R>
class ref_ptr
{
    R *ptr_;

  public:
    using ptr_type = R *;

    ref_ptr(ptr_type ptr) : ptr_(ptr) {}

    R *get() const { return ptr_; }
};

template <typename R>
class view_ptr
{
    const R *ptr_;

  public:
    using ptr_type = const R *;

    view_ptr(ptr_type ptr) : ptr_(ptr) {}

    const R *get() const { return ptr_; }
};

template <typename R, typename D>
struct basic_tensor_traits<R, owner, D> {
    using ptr_type = R *;
    using ref_type = R &;

    using Data = own_ptr<R, D>;
    using Access = readwrite;  // FIXME: use basic_access_traits
};

template <typename R, typename D>
struct basic_tensor_traits<R, readwrite, D> {
    using ptr_type = R *;
    using ref_type = R &;

    using Data = ref_ptr<R>;
    using Access = readwrite;
};

template <typename R, typename D>
struct basic_tensor_traits<R, readonly, D> {
    using ptr_type = const R *;
    using ref_type = const R &;

    using Data = view_ptr<R>;
    using Access = readonly;
};

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readwrite> ref(const basic_tensor<R, S, D, owner> &t)
{
    return basic_tensor<R, S, D, readwrite>(t);
}

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readonly> view(const basic_tensor<R, S, D, owner> &t)
{
    return basic_tensor<R, S, D, readwrite>(t);
}

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readonly> view(const basic_tensor<R, S, D, readwrite> &t)
{
    return basic_tensor<R, S, D, readonly>(t);
}

template <typename E, typename S, typename D>
basic_raw_tensor<E, S, D, readwrite>
ref(const basic_raw_tensor<E, S, D, owner> &t)
{
    return basic_raw_tensor<E, S, D, readwrite>(t);
}

template <typename E, typename S, typename D>
basic_raw_tensor<E, S, D, readonly>
view(const basic_raw_tensor<E, S, D, owner> &t)
{
    return basic_raw_tensor<E, S, D, readwrite>(t);
}

template <typename E, typename S, typename D>
basic_raw_tensor<E, S, D, readonly>
view(const basic_raw_tensor<E, S, D, readwrite> &t)
{
    return basic_raw_tensor<E, S, D, readonly>(t);
}
}  // namespace internal
}  // namespace ttl
