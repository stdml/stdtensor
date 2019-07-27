#pragma once
#include <ttl/bits/std_allocator.hpp>

namespace ttl
{
namespace internal
{
struct owner;
struct readwrite;
struct readonly;

template <typename R, typename A> struct basic_tensor_traits;

template <typename R> struct basic_tensor_traits<R, owner> {
    using ptr_type = R *;
    using ref_type = R &;

    using D = ref_ptr<R>;
    using IterA = readwrite;
};

template <typename R> struct basic_tensor_traits<R, readwrite> {
    using ptr_type = R *;
    using ref_type = R &;

    using D = ref_ptr<R>;
    using IterA = readwrite;
};

template <typename R> struct basic_tensor_traits<R, readonly> {
    using ptr_type = const R *;
    using ref_type = const R &;

    using D = view_ptr<R>;
    using IterA = readonly;
};
}  // namespace internal
}  // namespace ttl
