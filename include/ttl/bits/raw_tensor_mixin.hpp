#pragma once
#include <ttl/bits/raw_shape.hpp>

namespace ttl
{
namespace internal
{
template <typename Encoder, typename S, typename D, typename A>
class raw_tensor_mixin
{
  protected:
    using value_type_t = typename Encoder::value_type;

    const value_type_t value_type_;
    const S shape_;

    raw_tensor_mixin(const value_type_t value_type, const S &shape)
        : value_type_(value_type), shape_(shape)
    {
    }

  public:
    using encoder_type = Encoder;
    using shape_type = S;

    value_type_t value_type() const { return value_type_; }

    size_t data_size() const
    {
        return encoder_type::size(value_type_) * shape_.size();
    }

    const S &shape() const { return shape_; }
};
}  // namespace internal
}  // namespace ttl
