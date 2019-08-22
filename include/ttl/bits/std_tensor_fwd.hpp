#pragma once

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename shape_t> class basic_host_tensor;
template <typename R, rank_t r, typename shape_t> class basic_host_tensor_ref;
template <typename R, rank_t r, typename shape_t> class basic_host_tensor_view;

template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_ref;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_view;
}  // namespace internal
}  // namespace ttl
