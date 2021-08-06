#pragma once
#include <vector>

#include <stdml/bits/tensor/dtype.hpp>
#include <stdml/bits/tensor/shape.hpp>
#include <ttl/bits/std_def.hpp>

namespace stdml
{
class TenosrTupleType
{
    using tt = std::pair<DType, Shape>;
    using ttt = std::vector<tt>;

    ttt value_;

  public:
    TenosrTupleType();

    TenosrTupleType(DType dt, Shape shape);

    TenosrTupleType(ttt value);

    TenosrTupleType operator+(const TenosrTupleType &t) const;

    // FIXME: n * TTT and TTT * n are different
    TenosrTupleType operator*(size_t n) const;

    tt operator[](ttl::arity_t i) const;

    size_t arity() const;

    bool operator!=(const TenosrTupleType &t) const;
};
}  // namespace stdml
