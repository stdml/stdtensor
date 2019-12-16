#pragma once
#include <ttl/bits/raw_shape.hpp>
#include <ttl/bits/std_tensor_traits.hpp>

namespace ttl
{
namespace internal
{
template <typename A>
class raw_tensor_traits;

template <>
class raw_tensor_traits<owner>
{
  public:
    using ptr_type = void *;
    using Data = std::unique_ptr<char[]>;
};

template <>
class raw_tensor_traits<readwrite>
{
  public:
    using ptr_type = void *;
    using Data = ref_ptr<void>;
};

template <>
class raw_tensor_traits<readonly>
{
  public:
    using ptr_type = const void *;
    using Data = view_ptr<void>;
};

template <typename Encoder, typename S, typename D, typename A>
class raw_tensor_mixin
{
    using trait = raw_tensor_traits<A>;
    using data_ptr = typename trait::ptr_type;
    using data_t = typename trait::Data;

  protected:
    using value_type_t = typename Encoder::value_type;

    const value_type_t value_type_;
    const S shape_;
    data_t data_;

    raw_tensor_mixin(data_ptr data, const value_type_t value_type,
                     const S &shape)
        : value_type_(value_type), shape_(shape),
          data_(reinterpret_cast<char *>(const_cast<void *>(data)))
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
