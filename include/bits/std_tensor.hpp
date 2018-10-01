#pragma once
#include <memory>

#include <bits/std_shape.hpp>

template <typename R, rank_t r> struct basic_tensor;
template <typename R, rank_t r> struct basic_tensor_ref;
// template <typename R, rank_t r> struct basic_tensor_iterator;
// template <typename R, rank_t r> struct basic_tensor_view;

template <typename R, rank_t r> class basic_tensor
{
  public:
    template <typename... D>
    constexpr explicit basic_tensor(D... d)
        : shape_(d...), data_(new R[shape_.size()])
    {
    }

    template <typename... I> R &at(I... i) { return data_[shape_.index(i...)]; }

  private:
    const shape<r> shape_;
    const std::unique_ptr<R[]> data_;
};

template <typename R, rank_t r> class basic_tensor_ref
{
  public:
    template <typename... D>
    constexpr explicit basic_tensor_ref(R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    template <typename... I> R &at(I... i) { return data_[shape_.index(i...)]; }

  private:
    const shape<r> shape_;
    R *const data_;
};
