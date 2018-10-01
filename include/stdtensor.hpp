#pragma once
#include <bits/std_tensor.hpp>

namespace ttl
{
template <typename R, rank_t r> using tensor = basic_tensor<R, r>;
template <typename R, rank_t r> using tensor_ref = basic_tensor_ref<R, r>;
template <typename R, rank_t r> using tensor_view = basic_tensor_view<R, r>;

// Don't be confused with std::vector
template <typename R> using vector = tensor<R, 1>;
template <typename R> using vector_ref = tensor_ref<R, 1>;
template <typename R> using vector_view = tensor_view<R, 1>;

template <typename R> using matrix = tensor<R, 2>;
template <typename R> using matrix_ref = tensor_ref<R, 2>;
template <typename R> using matrix_view = tensor_view<R, 2>;
}  // namespace ttl

using namespace ttl;
