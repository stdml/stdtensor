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

template <typename shape_t = basic_raw_shape<>> class basic_raw_tensor
{
    const data_type_info value_type_;
    const shape_t shape_;
    std::unique_ptr<char[]> data_;

  public:
    template <typename... D>
    explicit basic_raw_tensor(const data_type_info &value_type, D... d)
        : value_type_(value_type), shape_(d...),
          data_(new char[shape_.size() * value_type.size])
    {
    }

    explicit basic_raw_tensor(const data_type_info &value_type,
                              const shape_t &shape)
        : value_type_(value_type), shape_(shape),
          data_(new char[shape_.size() * value_type.size])
    {
    }

    shape_t shape() const { return shape_; }

    template <typename R> R *data() const
    {
        return reinterpret_cast<R *>(data_.get());
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_ref<R, r, shape_type> ref_as() const
    {
        return _ranked_as<basic_tensor_ref<R, r, shape_type>>();
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_view<R, r, shape_type> view_as() const
    {
        return _ranked_as<basic_tensor_view<R, r, shape_type>>();
    }

  private:
    template <typename T> T _ranked_as() const
    {
        using R = typename T::value_type;
        // TODO: use contracts of c++20
        if (typeinfo<R>().code != value_type_.code) {
            throw std::invalid_argument("invalid scalar type");
        }
        return T(reinterpret_cast<R *>(data_.get()),
                 shape_.template as_ranked<T::rank>());
    }
};

}  // namespace internal
}  // namespace ttl
