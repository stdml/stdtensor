#pragma once
#include <bits/std_flat_tensor.hpp>
#include <bits/std_raw_tensor.hpp>
#include <bits/std_scalar_type_encoding.hpp>
#include <bits/std_tensor.hpp>
#include <bits/std_type_encoder.hpp>

namespace ttl
{
template <typename R, internal::rank_t r>
using tensor = internal::basic_tensor<R, r>;

template <typename R, internal::rank_t r>
using tensor_ref = internal::basic_tensor_ref<R, r>;

template <typename R, internal::rank_t r>
using tensor_view = internal::basic_tensor_view<R, r>;

// Don't be confused with std::vector
template <typename R> using vector = tensor<R, 1>;
template <typename R> using vector_ref = tensor_ref<R, 1>;
template <typename R> using vector_view = tensor_view<R, 1>;

template <typename R> using matrix = tensor<R, 2>;
template <typename R> using matrix_ref = tensor_ref<R, 2>;
template <typename R> using matrix_view = tensor_view<R, 2>;

using raw_tensor = internal::basic_raw_tensor<
    internal::basic_type_encoder<internal::idx_format::encoding>>;

template <typename R> using flat_tensor = internal::basic_flat_tensor<R>;
}  // namespace ttl
