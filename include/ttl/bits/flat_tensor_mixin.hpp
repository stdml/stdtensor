#pragma once
#include <ttl/bits/raw_shape.hpp>
#include <ttl/bits/std_shape.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>
#include <ttl/bits/std_tensor_traits.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D, typename A>
class flat_tensor_mixin
{
    using trait = basic_tensor_traits<R, A, D>;
    using data_ptr = typename trait::ptr_type;
    using data_ref = typename trait::ref_type;
    using data_t = typename trait::Data;

    using Dim = typename S::dimension_type;

    const S shape_;
    data_t data_;

    // FIXME: support slice and iterator
    // using slice_type = basic_tensor<R, S, D, typename trait::Access>;

    template <rank_t r, typename A1 = typename trait::Access>
    using T = basic_tensor<R, basic_shape<r, Dim>, D, A1>;

    template <rank_t r, typename A1>
    T<r, A1> ranked_as() const
    {
        return T<r, A1>(data_.get(), shape_.template as_ranked<r>());
    }

  protected:
    using allocator = basic_allocator<R, D>;

    explicit flat_tensor_mixin(data_ptr data, const S &shape)
        : shape_(shape), data_(data)
    {
    }

  public:
    using value_type = R;
    using shape_type = S;
    using device_type = D;

    size_t data_size() const { return shape_.size() * sizeof(R); }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + shape_.size(); }

    template <rank_t r>
    T<r, readwrite> ref_as() const
    {
        return ranked_as<r, readwrite>();
    }

    template <rank_t r>
    T<r, readonly> view_as() const
    {
        return ranked_as<r, readonly>();
    }
};
}  // namespace internal
}  // namespace ttl
