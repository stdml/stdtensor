#pragma once
#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/std_access_traits.hpp>
#include <ttl/bits/std_except.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>
#include <ttl/bits/std_tensor_traits.hpp>

namespace ttl
{
namespace internal
{
template <typename A, typename D>
class raw_tensor_traits;

template <typename D>
class raw_tensor_traits<owner, D>
{
  public:
    using ptr_type = void *;
    using Data = own_ptr<char, D>;
};

template <typename D>
class raw_tensor_traits<readwrite, D>
{
  public:
    using ptr_type = void *;
    using Data = ref_ptr<void>;
};

template <typename D>
class raw_tensor_traits<readonly, D>
{
  public:
    using ptr_type = const void *;
    using Data = view_ptr<void>;
};

template <typename Encoder, typename S, typename D, typename A>
class raw_tensor_mixin
{
    using trait = raw_tensor_traits<A, D>;
    using data_ptr = typename trait::ptr_type;
    using data_t = typename trait::Data;

    using value_type_t = typename Encoder::value_type;

    const value_type_t value_type_;
    const S shape_;
    data_t data_;

  protected:
    using Dim = typename S::dimension_type;  // For MSVC C2248

    raw_tensor_mixin(data_ptr data, const value_type_t value_type,
                     const S &shape)
        : value_type_(value_type), shape_(shape),
          data_(reinterpret_cast<char *>(const_cast<void *>(data)))
    {
    }

  public:
    using encoder_type = Encoder;
    using shape_type = S;

    template <typename R>
    static constexpr value_type_t type()
    {
        return Encoder::template value<R>();
    }

    value_type_t value_type() const { return value_type_; }

    size_t data_size() const
    {
        return Encoder::size(value_type_) * shape_.size();
    }

    rank_t rank() const { return shape_.rank(); }

    Dim size() const { return shape_.size(); }

    const auto &dims() const { return shape_.dims(); }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return (char *)(data_.get()) + data_size(); }

    template <typename R>
    typename basic_tensor_traits<R, A, D>::ptr_type data() const
    {
        if (type<R>() != value_type_) {
            throw invalid_type_reification(typeid(R));
        }
        using ptr_type = typename basic_tensor_traits<R, A, D>::ptr_type;
        return reinterpret_cast<ptr_type>(data_.get());
    }

    template <typename R>
    auto typed() const
    {
        using Access = typename basic_access_traits<A>::type;
        using T = basic_tensor<R, basic_flat_shape<Dim>, D, Access>;
        return T(data<R>(), shape_);
    }

    template <typename R, rank_t r>
    auto typed() const
    {
        using Access = typename basic_access_traits<A>::type;
        using T = basic_tensor<R, basic_shape<r, Dim>, D, Access>;
        return T(data<R>(), shape_.template ranked<r>());
    }
};
}  // namespace internal
}  // namespace ttl
