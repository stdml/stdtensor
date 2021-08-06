#include <stdml/bits/tensor/tuple_type.hpp>

namespace stdml
{
TenosrTupleType::TenosrTupleType() {}

TenosrTupleType::TenosrTupleType(DType dt, Shape shape)
    : value_({std::make_pair(dt, std::move(shape))})
{
}

TenosrTupleType::TenosrTupleType(ttt value) : value_(std::move(value)) {}

TenosrTupleType TenosrTupleType::operator+(const TenosrTupleType &t) const
{
    ttt value = value_;
    value.insert(value.end(), t.value_.begin(), t.value_.end());
    return TenosrTupleType(std::move(value));
}

TenosrTupleType TenosrTupleType::operator*(size_t n) const
{
    ttt value;
    value.reserve(value_.size());
    std::transform(value_.begin(), value_.end(), std::back_inserter(value),
                   [n = n](auto p) {
                       auto dims = p.second.dims();
                       dims.insert(dims.begin(), n);
                       return std::make_pair(p.first, Shape(dims));
                   });
    return TenosrTupleType(value);
}

bool TenosrTupleType::operator!=(const TenosrTupleType &t) const
{
    return value_ != t.value_;
}

size_t TenosrTupleType::arity() const { return value_.size(); }
}  // namespace stdml
