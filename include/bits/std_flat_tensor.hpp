#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <bits/std_raw_shape.hpp>
#include <bits/std_scalar_type_encoding.hpp>
#include <bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{

template <typename R, typename shape_t = basic_raw_shape<>>
class basic_flat_tensor
{
    const shape_t shape_;
    std::unique_ptr<R[]> data_;

  public:
    using value_type = R;

    template <typename... D>
    explicit basic_flat_tensor(D... d) : basic_flat_tensor(shape_t(d...))
    {
    }

    explicit basic_flat_tensor(const shape_t &shape)
        : shape_(shape), data_(new R[shape_.size()])
    {
    }

    shape_t shape() const { return shape_; }

    R *data() const { return data_.get(); }

    template <rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_ref<R, r, shape_type> ref_as() const
    {
        return ranked_as<basic_tensor_ref<R, r, shape_type>>();
    }

    template <rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_view<R, r, shape_type> view_as() const
    {
        return ranked_as<basic_tensor_view<R, r, shape_type>>();
    }

  private:
    template <typename T> T ranked_as() const
    {
        static_assert(std::is_same<R, typename T::value_type>::value, "");
        return T(data_.get(), shape_.template as_ranked<T::rank>());
    }
};

}  // namespace internal
}  // namespace ttl
