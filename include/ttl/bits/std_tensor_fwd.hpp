#pragma once

namespace ttl
{
namespace internal
{
template <typename R, typename D>
class basic_allocator;

template <typename R, typename D>
class basic_deallocator;

template <typename D1, typename D2>
class basic_copier;

template <typename R, typename A, typename D>
struct basic_tensor_traits;

template <typename R, typename S, typename D, typename A>
class basic_scalar_mixin;

template <typename R, typename S, typename D, typename A>
class basic_tensor_mixin;

template <typename R, typename S, typename D, typename A>
class basic_tensor;

template <typename Encoder, typename S, typename D, typename A>
class basic_raw_tensor;
}  // namespace internal
}  // namespace ttl
