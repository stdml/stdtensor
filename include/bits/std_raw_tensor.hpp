#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <bits/std_raw_shape.hpp>
#include <bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{

template <typename DataEncoder, typename shape_t = basic_raw_shape<>>
class basic_raw_tensor
{
    using value_type_t = typename DataEncoder::value_type;

    const value_type_t value_type_;
    const shape_t shape_;
    std::unique_ptr<char[]> data_;

  public:
    using encoder_type = DataEncoder;
    using shape_type = shape_t;

    template <typename... D>
    explicit basic_raw_tensor(const value_type_t value_type, D... d)
        : basic_raw_tensor(value_type, shape_t(d...))
    {
    }

    explicit basic_raw_tensor(const value_type_t value_type,
                              const shape_t &shape)
        : value_type_(value_type), shape_(shape),
          data_(new char[shape_.size() * DataEncoder::size(value_type)])
    {
    }

    shape_t shape() const { return shape_; }

    template <typename R> R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<R *>(data_.get());
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_ref<R, r, shape_type> ref_as() const
    {
        return ranked_as<basic_tensor_ref<R, r, shape_type>>();
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_tensor_view<R, r, shape_type> view_as() const
    {
        return ranked_as<basic_tensor_view<R, r, shape_type>>();
    }

  private:
    template <typename T> T ranked_as() const
    {
        return T(data<typename T::value_type>(),
                 shape_.template as_ranked<T::rank>());
    }
};

}  // namespace internal
}  // namespace ttl
