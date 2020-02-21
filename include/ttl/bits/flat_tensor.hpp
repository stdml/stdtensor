#pragma once
#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/flat_tensor_mixin.hpp>
#include <ttl/bits/std_tensor.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_flat_shape<Dim>, D, owner>
    : public flat_tensor_mixin<R, basic_flat_shape<Dim>, D, owner>
{
    using S = basic_flat_shape<Dim>;
    using mixin = flat_tensor_mixin<R, S, D, owner>;

  public:
    template <typename... Dims>
    constexpr explicit basic_tensor(Dims... d) : basic_tensor(S(d...))
    {
    }

    constexpr explicit basic_tensor(const S &shape)
        : mixin(typename mixin::allocator()(shape.size()), shape)
    {
    }
};

template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_flat_shape<Dim>, D, readwrite>
    : public flat_tensor_mixin<R, basic_flat_shape<Dim>, D, readwrite>
{
    using S = basic_flat_shape<Dim>;
    using mixin = flat_tensor_mixin<R, S, D, readwrite>;

  public:
    template <typename... Dims>
    constexpr explicit basic_tensor(R *data, Dims... d)
        : basic_tensor(data, S(d...))
    {
    }

    basic_tensor(const basic_tensor<R, S, D, owner> &t)
        : basic_tensor(t.data(), t.shape())
    {
    }

    constexpr explicit basic_tensor(R *data, const S &shape)
        : mixin(data, shape)
    {
    }
};

template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_flat_shape<Dim>, D, readonly>
    : public flat_tensor_mixin<R, basic_flat_shape<Dim>, D, readonly>
{
    using S = basic_flat_shape<Dim>;
    using mixin = flat_tensor_mixin<R, S, D, readonly>;

  public:
    template <typename... Dims>
    constexpr explicit basic_tensor(const R *data, Dims... d)
        : basic_tensor(data, S(d...))
    {
    }

    basic_tensor(const basic_tensor<R, S, D, owner> &t)
        : basic_tensor(t.data(), t.shape())
    {
    }

    basic_tensor(const basic_tensor<R, S, D, readwrite> &t)
        : basic_tensor(t.data(), t.shape())
    {
    }

    constexpr explicit basic_tensor(const R *data, const S &shape)
        : mixin(data, shape)
    {
    }
};

template <typename R, typename D>
using basic_flat_tensor = basic_tensor<R, basic_flat_shape<>, D, owner>;

template <typename R, typename D>
using basic_flat_tensor_ref = basic_tensor<R, basic_flat_shape<>, D, readwrite>;

template <typename R, typename D>
using basic_flat_tensor_view = basic_tensor<R, basic_flat_shape<>, D, readonly>;
}  // namespace internal
}  // namespace ttl
