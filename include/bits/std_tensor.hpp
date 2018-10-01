#pragma once
#include <memory>

#include <bits/std_shape.hpp>

/* forward declarations */

template <typename R, rank_t r, typename shape_t> struct basic_tensor;
template <typename R, rank_t r, typename shape_t> struct basic_tensor_ref;
// template <typename R, rank_t r> struct basic_tensor_iterator;
template <typename R, rank_t r, typename shape_t> struct basic_tensor_view;

/* specialization for rank 0 */

template <typename R, typename shape_t> class basic_tensor_ref<R, 0, shape_t>
{
  public:
    basic_tensor_ref(R *data) : data_(data) {}

    basic_tensor_ref(R *data, const shape_t &) : data_(data) {}

    //   private:
    R *const data_;
};

template <typename R, typename shape_t> class basic_tensor_view<R, 0, shape_t>
{
    // TODO
};

template <typename R, typename shape_t>
R &scalar(const basic_tensor_ref<R, 0, shape_t> &t)
{
    return t.data_[0];
}

/* rank > 0 */

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor
{
    // using subspace_ref_t =
    //     basic_tensor_ref<R, r - 1, typename shape_t::template subshape_t<1>>;

  public:
    template <typename... D>
    constexpr explicit basic_tensor(D... d)
        : shape_(d...), data_(new R[shape_.size()])
    {
    }

    R *data() { return data_.get(); }

    const R *data() const { return data_.get(); }

    shape_t shape() const { return shape_; }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
    }

    // subspace_ref_t operator[](int i) const { return ref(*this)[i]; }

  private:
    const shape_t shape_;
    const std::unique_ptr<R[]> data_;
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_ref
{
    using subspace_t =
        basic_tensor_ref<R, r - 1, typename shape_t::template subshape_t<1>>;

  public:
    template <typename... D>
    constexpr explicit basic_tensor_ref(R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    constexpr explicit basic_tensor_ref(R *data, const shape_t &shape)
        : shape_(shape), data_(data)
    {
    }

    R *data() { return data_; }

    const R *data() const { return data_; }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
    }

    subspace_t operator[](int i) const
    {
        return subspace_t(data_ + i * shape_.subspace_size(),
                          shape_.subshape());
    }

  private:
    const shape_t shape_;
    R *const data_;
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_view
{
  public:
    template <typename... D>
    constexpr explicit basic_tensor_view(const R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    template <typename... I> R at(I... i) { return data_[shape_.offset(i...)]; }

  private:
    const shape_t shape_;
    const R *const data_;
};
