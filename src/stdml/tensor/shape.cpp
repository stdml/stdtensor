#include <algorithm>
#include <experimental/iterator>
#include <iostream>
#include <sstream>

#include <stdml/bits/tensor/shape.hpp>

namespace stdml
{
std::ostream &operator<<(std::ostream &os, const Shape &s)
{
    os << '(';
    std::copy(s.dims().begin(), s.dims().end(),
              std::experimental::make_ostream_joiner(os, ", "));
    os << ')';
    return os;
}
}  // namespace stdml
