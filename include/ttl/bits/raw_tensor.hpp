#pragma once
#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/flat_tensor.hpp>
#include <ttl/bits/raw_tensor_mixin.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename Encoder, typename Dim, typename D>
class basic_raw_tensor<Encoder, basic_flat_shape<Dim>, D, owner>
    : public raw_tensor_mixin<Encoder, basic_flat_shape<Dim>, D, owner>
{
    using value_type_t = typename Encoder::value_type;
    using S = basic_flat_shape<Dim>;
    using mixin = raw_tensor_mixin<Encoder, S, D, owner>;
    using allocator = basic_allocator<char, D>;

  public:
    template <typename... Dims>
    explicit basic_raw_tensor(const value_type_t value_type, Dims... d)
        : basic_raw_tensor(value_type, S(d...))
    {
    }

    explicit basic_raw_tensor(const value_type_t value_type, const S &shape)
        : mixin(allocator()(shape.size() * Encoder::size(value_type)),
                value_type, shape)
    {
    }
};

template <typename Encoder, typename Dim, typename D>
class basic_raw_tensor<Encoder, basic_flat_shape<Dim>, D, readwrite>
    : public raw_tensor_mixin<Encoder, basic_flat_shape<Dim>, D, readwrite>
{
    using value_type_t = typename Encoder::value_type;
    using S = basic_flat_shape<Dim>;
    using mixin = raw_tensor_mixin<Encoder, S, D, readwrite>;

  public:
    template <typename... Dims>
    explicit basic_raw_tensor(void *data, const value_type_t value_type,
                              Dims... d)
        : basic_raw_tensor(data, value_type, S(d...))
    {
    }

    explicit basic_raw_tensor(void *data, const value_type_t value_type,
                              const S &shape)
        : mixin(data, value_type, shape)
    {
    }

    explicit basic_raw_tensor(const basic_raw_tensor<Encoder, S, D, owner> &t)
        : basic_raw_tensor(t.data(), t.value_type(), t.shape())
    {
    }

    template <typename R, typename S1>
    explicit basic_raw_tensor(const basic_tensor<R, S1, D, readwrite> &t)
        : basic_raw_tensor(t.data(), Encoder::template value<R>(), S(t.shape()))
    {
    }
};

template <typename Encoder, typename Dim, typename D>
class basic_raw_tensor<Encoder, basic_flat_shape<Dim>, D, readonly>
    : public raw_tensor_mixin<Encoder, basic_flat_shape<Dim>, D, readonly>
{
    using value_type_t = typename Encoder::value_type;
    using S = basic_flat_shape<Dim>;
    using mixin = raw_tensor_mixin<Encoder, S, D, readonly>;

  public:
    template <typename... Dims>
    explicit basic_raw_tensor(const void *data, const value_type_t value_type,
                              Dims... d)
        : basic_raw_tensor(data, value_type, S(d...))
    {
    }

    explicit basic_raw_tensor(const void *data, const value_type_t value_type,
                              const S &shape)
        : mixin(data, value_type, shape)
    {
    }

    explicit basic_raw_tensor(const basic_raw_tensor<Encoder, S, D, owner> &t)
        : mixin(t.data(), t.value_type(), t.shape())
    {
    }

    explicit basic_raw_tensor(
        const basic_raw_tensor<Encoder, S, D, readwrite> &t)
        : mixin(t.data(), t.value_type(), t.shape())
    {
    }

    template <typename R, typename S1>
    explicit basic_raw_tensor(const basic_tensor<R, S1, D, readonly> &t)
        : mixin(t.data(), Encoder::template value<R>(), S(t.shape()))
    {
    }
};

template <typename E, typename D>
using raw_tensor = basic_raw_tensor<E, basic_flat_shape<>, D, owner>;

template <typename E, typename D>
using raw_tensor_ref = basic_raw_tensor<E, basic_flat_shape<>, D, readwrite>;

template <typename E, typename D>
using raw_tensor_view = basic_raw_tensor<E, basic_flat_shape<>, D, readonly>;
}  // namespace internal
}  // namespace ttl
