#pragma once
#include <ttl/bits/flat_shape.hpp>
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

    const S shape_;
    data_t data_;

    // FIXME: support slice and iterator
    // using slice_type = basic_tensor<R, S, D, typename trait::Access>;

    template <rank_t r, typename A1 = typename trait::Access>
    using T =
        basic_tensor<R, basic_shape<r, typename S::dimension_type>, D, A1>;

  protected:
    using Dim = typename S::dimension_type;  // For MSVC C2248
    using allocator = basic_allocator<R, D>;

    explicit flat_tensor_mixin(data_ptr data, const S &shape)
        : shape_(shape), data_(data)
    {
    }

  public:
    using value_type = R;
    using shape_type = S;
    using device_type = D;

    rank_t rank() const { return shape_.rank(); }

    Dim size() const { return shape_.size(); }

    const auto &dims() const { return shape_.dims(); }

    size_t data_size() const { return shape_.size() * sizeof(R); }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + shape_.size(); }

    template <rank_t r>
    auto ranked() const
    {
        using Access = typename basic_access_traits<A>::type;
        using T = basic_tensor<R, basic_shape<r, Dim>, D, Access>;
        return T(data(), shape_.template ranked<r>());
    }
};
}  // namespace internal
}  // namespace ttl
