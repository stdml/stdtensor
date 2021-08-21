#include <algorithm>
#include <experimental/iterator>
#include <iostream>
#include <sstream>

#include <stdml/bits/tensor/shape.hpp>

namespace stdml
{
Shape::Shape(S s) : s_(std::move(s)) {}

ttl::rank_t Shape::rank() const { return s_.rank(); }

size_t Shape::size() const { return s_.size(); }

bool Shape::operator==(const Shape &s) const { return s_ == s.s_; }

Shape Shape::subshape() const { return s_.subshape(); }

std::ostream &operator<<(std::ostream &os, const Shape &s)
{
    os << '(';
    std::copy(s.dims().begin(), s.dims().end(),
              std::experimental::make_ostream_joiner(os, ", "));
    os << ')';
    return os;
}
}  // namespace stdml
