#pragma once
#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/raw_tensor.hpp>
#include <ttl/bits/std_encoding.hpp>
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/bits/type_encoder.hpp>

namespace stdml
{
using dim_t = int64_t;

using idx_encoder =
    ttl::internal::basic_type_encoder<ttl::experimental::std_encoding>;

using flat_shape = ttl::internal::basic_flat_shape<dim_t>;

template <typename D, typename A>
using raw_tensor_t =
    ttl::internal::basic_raw_tensor<idx_encoder, flat_shape, D, A>;

using raw_tensor =
    raw_tensor_t<ttl::internal::host_memory, ttl::internal::owner>;
using raw_tensor_ref =
    raw_tensor_t<ttl::internal::host_memory, ttl::internal::readwrite>;
using raw_tensor_view =
    raw_tensor_t<ttl::internal::host_memory, ttl::internal::readonly>;
}  // namespace stdml
