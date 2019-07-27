#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_raw_shape.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{

template <typename R, typename shape_t, typename data_holder_t>
class abstract_flat_tensor
{
    const shape_t shape_;
    data_holder_t data_;

  protected:
    explicit abstract_flat_tensor(const shape_t shape, R *data)
        : shape_(shape), data_(data)
    {
    }

  public:
    using value_type = R;
    using shape_type = shape_t;

    shape_t shape() const { return shape_; }

    R *data() const { return data_.get(); }

    R *data_end() const { return data_.get() + shape().size(); }

    template <rank_t r, typename shape_type = basic_shape<r>>
    basic_host_tensor_ref<R, r, shape_type> ref_as() const
    {
        return ranked_as<basic_host_tensor_ref<R, r, shape_type>>();
    }

    template <rank_t r, typename shape_type = basic_shape<r>>
    basic_host_tensor_view<R, r, shape_type> view_as() const
    {
        return ranked_as<basic_host_tensor_view<R, r, shape_type>>();
    }

  private:
    template <typename T> T ranked_as() const
    {
        static_assert(std::is_same<R, typename T::value_type>::value, "");
        return T(data_.get(), shape_.template as_ranked<T::rank>());
    }
};

template <typename R, typename shape_t = basic_raw_shape<>>
class basic_flat_tensor : public abstract_flat_tensor<R, shape_t, own_ptr<R>>
{
  public:
    template <typename... D>
    explicit basic_flat_tensor(D... d) : basic_flat_tensor(shape_t(d...))
    {
    }

    explicit basic_flat_tensor(const shape_t &shape)
        : abstract_flat_tensor<R, shape_t, own_ptr<R>>(shape,
                                                       new R[shape.size()])
    {
    }
};

template <typename R, typename shape_t = basic_raw_shape<>>
class basic_flat_tensor_ref
    : public abstract_flat_tensor<R, shape_t, ref_ptr<R>>
{
  public:
    template <typename... D>
    explicit basic_flat_tensor_ref(R *data, D... d)
        : basic_flat_tensor_ref(data, shape_t(d...))
    {
    }

    explicit basic_flat_tensor_ref(R *data, const shape_t &shape)
        : abstract_flat_tensor<R, shape_t, ref_ptr<R>>(shape, data)
    {
    }
};

template <typename R, typename shape_t>
basic_flat_tensor_ref<R, shape_t> ref(const basic_flat_tensor<R, shape_t> &t)
{
    return basic_flat_tensor_ref<R, shape_t>(t.data(), t.shape());
}

template <typename R, typename shape_t = basic_raw_shape<>>
class basic_flat_tensor_view
    : public abstract_flat_tensor<R, shape_t, view_ptr<R>>
{
  public:
    template <typename... D>
    explicit basic_flat_tensor_view(const R *data, D... d)
        : basic_flat_tensor_view(data, shape_t(d...))
    {
    }

    explicit basic_flat_tensor_view(const R *data, const shape_t &shape)
        : abstract_flat_tensor<R, shape_t, view_ptr<R>>(shape, data)
    {
    }
};

}  // namespace internal
}  // namespace ttl
