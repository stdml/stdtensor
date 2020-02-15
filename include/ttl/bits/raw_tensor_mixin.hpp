#pragma once
#include <stdexcept>
#include <ttl/bits/raw_shape.hpp>
#include <ttl/bits/std_access_traits.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>
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

    using value_type_t = typename Encoder::value_type;

    const value_type_t value_type_;
    const S shape_;
    data_t data_;

    using Dim = typename S::dimension_type;

  protected:
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
        return Encoder::size(value_type_) * shape_.size();
    }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return (char *)(data_.get()) + data_size(); }

    template <typename R>
    typename basic_tensor_traits<R, A, D>::ptr_type data() const
    {
        // TODO: use contracts of c++20
        if (Encoder::template value<R>() != value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        using ptr_type = typename basic_tensor_traits<R, A, D>::ptr_type;
        return reinterpret_cast<ptr_type>(data_.get());
    }

    template <typename R>
    auto typed() const
    {
        using Access = typename basic_access_traits<A>::type;
        using T = basic_tensor<R, basic_raw_shape<Dim>, D, Access>;
        return T(data<R>(), shape_);
    }

    template <typename R, rank_t r, typename A1 = A>
    basic_tensor<R, basic_shape<r, Dim>, D, A1> ranked_as() const
    {
        return basic_tensor<R, basic_shape<r, Dim>, D, A1>(
            data<R>(), shape_.template as_ranked<r>());
    }

    template <typename R, rank_t r>
    basic_tensor<R, basic_shape<r, Dim>, D, readwrite> ref_as() const
    {
        return ranked_as<R, r, readwrite>();
    }

    template <typename R, rank_t r>
    basic_tensor<R, basic_shape<r, Dim>, D, readonly> view_as() const
    {
        return ranked_as<R, r, readonly>();
    }
};
}  // namespace internal
}  // namespace ttl
