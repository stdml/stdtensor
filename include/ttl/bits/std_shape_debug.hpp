#pragma once
#include <experimental/iterator>
#include <sstream>

#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
template <typename Dims>
std::string _join_string(const Dims &dims, const std::string &sep = ", ",
                         const std::string &bgn = "(",
                         const std::string &end = ")")
{
    std::stringstream ss;
    ss << bgn;
    std::copy(dims.begin(), dims.end(),
              std::experimental::make_ostream_joiner(ss, sep));
    ss << end;
    return ss.str();
}

template <rank_t r, typename Dim>
std::string to_string(const basic_shape<r, Dim> &shape)
{
    return _join_string(shape.dims());
}

template <typename Dim>
std::string to_string(const basic_flat_shape<Dim> &shape)
{
    return _join_string(shape.dims());
}
}  // namespace internal
}  // namespace ttl
