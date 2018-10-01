#pragma once
#include <memory>

#include <bits/std_shape.hpp>

template <typename R, rank_t r, typename shape_t> struct basic_tensor;
template <typename R, rank_t r, typename shape_t> struct basic_tensor_ref;
// template <typename R, rank_t r> struct basic_tensor_iterator;
template <typename R, rank_t r, typename shape_t> struct basic_tensor_view;

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor
{
  public:
    template <typename... D>
    constexpr explicit basic_tensor(D... d)
        : shape_(d...), data_(new R[shape_.size()])
    {
    }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
    }

  private:
    const shape_t shape_;
    const std::unique_ptr<R[]> data_;
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_ref
{
  public:
    template <typename... D>
    constexpr explicit basic_tensor_ref(R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
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
