#pragma once
#include <cassert>
#include <cstdint>
#include <memory>

#include <bits/std_data_type_encoding.hpp>
#include <bits/std_generic_shape.hpp>
#include <bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{

template <typename shape_t = basic_generic_shape<std::uint32_t>>
class basic_generic_tensor
{
    const data_type_info value_type_;
    const shape_t shape_;
    std::unique_ptr<char[]> data_;

  public:
    template <typename... D>
    explicit basic_generic_tensor(const data_type_info &value_type, D... d)
        : value_type_(value_type), shape_(d...),
          data_(new char[shape_.size() * value_type.size])
    {
    }

    explicit basic_generic_tensor(const data_type_info &value_type,
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
        constexpr rank_t r = T::rank;
        using R = typename T::value_type;
        using dim_t = typename shape_t::dim_type;

        // TODO: use contracts of c++20
        assert(typeinfo<R>().code == value_type_.code);
        assert(r == shape_.rank());
        std::array<dim_t, r> d;
        std::copy(shape_.dims.begin(), shape_.dims.end(), d.begin());
        const basic_shape<r, dim_t> shape(
            d);  // FIXME: use shape_.as_ranked<r>()
        return T(reinterpret_cast<R *>(data_.get()), shape);
    }
};

}  // namespace internal
}  // namespace ttl
