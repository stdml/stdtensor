#pragma once
#include <sstream>

#include <stdml/bits/tensor/tensor.hpp>
#include <ttl/bits/std_shape_debug.hpp>

namespace stdml
{
extern void show_tensor(std::basic_ostream<char> &os, const TensorView &x);

void copy(const TensorRef &y, const TensorView &x);

template <typename TT>
std::string info(const TT &x)
{
    std::stringstream ss;
    ss << tn(x.dtype()) << ttl::internal::to_string(x.shape().get()) << '@'
       << device_name(x.device());
    return ss.str();
}

template <typename TT>
std::string show(const TT &x)
{
    std::stringstream ss;
    ss << tn(x.dtype()) << ttl::internal::to_string(x.shape().get());
    show_tensor(ss, x);
    return ss.str();
}
}  // namespace stdml

#include <ttl/bits/std_host_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r, typename Dim>
std::string show(const basic_host_tensor_view<R, r, Dim> &t)
{
    stdml::TensorView x(t);
    return stdml::show(x);
}
}  // namespace internal

using internal::show;
}  // namespace ttl
