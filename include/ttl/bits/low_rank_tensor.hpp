#pragma once
#include <algorithm>
#include <array>
#include <utility>

#include <ttl/bits/std_access_traits.hpp>
#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_static_shape.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename Dim, Dim... ds>
class basic_tensor<R, basic_static_shape<Dim, ds...>, host_memory, owner>
{
  protected:
    using S = basic_static_shape<Dim, ds...>;
    using data_t = std::array<R, S::dim>;

    data_t data_;

    template <typename F>
    static void pointwise(const F &f, const data_t &x, data_t &y)
    {
        std::transform(x.begin(), x.end(), y.begin(), f);
    }

    template <typename F>
    static void pointwise(const F &f, const data_t &x, const data_t &y,
                          data_t &z)
    {
        std::transform(x.begin(), x.end(), y.begin(), z.begin(), f);
    }

  public:
    using shape_type = S;
    using dimension_type = Dim;

    static constexpr auto rank = 1;
    static constexpr Dim dim = S::dim;

    basic_tensor(void *)
    {
        // no initialization
    }

    basic_tensor() { std::fill(data_.begin(), data_.end(), static_cast<R>(0)); }

    explicit basic_tensor(data_t data) : data_(std::move(data)) {}

    template <typename... C>
    basic_tensor(const C &... c) : data_({static_cast<R>(c)...})
    {
        static_assert(sizeof...(C) == dim, "");
    }

    R *data() { return data_.data(); }

    const R *data() const { return data_.data(); }

    const R &operator[](int i) const { return data_[i]; }

    basic_tensor<R, S, host_memory, owner> operator-() const
    {
        data_t y;
        pointwise(std::negate<R>(), data_, y);
        return basic_tensor<R, S, host_memory, owner>(std::move(y));
    }

    basic_tensor<R, S, host_memory, owner> operator*(R x) const
    {
        data_t y;
        pointwise([x = x](R a) { return a * x; }, data_, y);
        return basic_tensor<R, S, host_memory, owner>(std::move(y));
    }

    template <typename A>
    basic_tensor<R, S, host_memory, owner>
    operator+(const basic_tensor<R, S, host_memory, A> &y) const
    {
        data_t z;
        pointwise(std::plus<R>(), data_, y.data_, z);
        return basic_tensor<R, S, host_memory, owner>(std::move(z));
    }

    template <typename A>
    basic_tensor<R, S, host_memory, owner>
    operator-(const basic_tensor<R, S, host_memory, A> &y) const
    {
        data_t z;
        pointwise(std::minus<R>(), data_, y.data_, z);
        return basic_tensor<R, S, host_memory, owner>(std::move(z));
    }
};
}  // namespace internal
}  // namespace ttl
