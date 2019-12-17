#pragma once
#include <experimental/iterator>
#include <sstream>

#include <ttl/bits/raw_shape.hpp>
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
template <rank_t r, typename D>
std::string to_string(const basic_shape<r, D> &shape)
{
    std::stringstream ss;
    ss << "(";
    std::copy(shape.dims().begin(), shape.dims().end(),
              std::experimental::make_ostream_joiner(ss, ", "));
    ss << ")";
    return ss.str();
}

template <typename D>
std::string to_string(const basic_raw_shape<D> &shape)
{
    std::stringstream ss;
    ss << "(";
    std::copy(shape.dims().begin(), shape.dims().end(),
              std::experimental::make_ostream_joiner(ss, ", "));
    ss << ")";
    return ss.str();
}

}  // namespace internal
}  // namespace ttl
