#pragma once

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_ref;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_view;

template <typename R, typename D> class basic_allocator;

template <typename R, typename S, typename D, typename A>
class basic_scalar_mixin;

template <typename R, typename S, typename D, typename A>
class basic_tensor_mixin;

template <typename R, typename S, typename D, typename A> class basic_tensor;
}  // namespace internal
}  // namespace ttl
