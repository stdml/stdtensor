#pragma once
#include <ttl/bits/std_shape.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>
#include <ttl/bits/std_tensor_mixin.hpp>

namespace ttl
{
namespace internal
{
// rank = 0
template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_shape<0, Dim>, D, owner>
    : public basic_scalar_mixin<R, basic_shape<0, Dim>, D, owner>
{
    using mixin = basic_scalar_mixin<R, basic_shape<0, Dim>, D, owner>;
    using allocator = basic_allocator<R, D>;

    own_ptr<R, D> data_owner_;

    basic_tensor(R *data) : mixin(data), data_owner_(data) {}

  public:
    basic_tensor() : basic_tensor(allocator()(1)) {}

    basic_tensor(const basic_shape<0, Dim> &) : basic_tensor() {}

    R operator=(const R &val) const { return *this->data() = val; }

    operator R &() const { return *this->data(); }
};

template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_shape<0, Dim>, D, readwrite>
    : public basic_scalar_mixin<R, basic_shape<0, Dim>, D, readwrite>
{
    using mixin = basic_scalar_mixin<R, basic_shape<0, Dim>, D, readwrite>;
    using mixin::mixin;

  public:
    basic_tensor(const basic_tensor<R, basic_shape<0, Dim>, D, owner> &t)
        : mixin(t.data())
    {
    }

    R operator=(const R &val) const { return *this->data() = val; }

    operator R &() const { return *this->data(); }
};

template <typename R, typename Dim, typename D>
class basic_tensor<R, basic_shape<0, Dim>, D, readonly>
    : public basic_scalar_mixin<R, basic_shape<0, Dim>, D, readonly>
{
    using mixin = basic_scalar_mixin<R, basic_shape<0, Dim>, D, readonly>;
    using mixin::mixin;

  public:
    basic_tensor(const basic_tensor<R, basic_shape<0, Dim>, D, owner> &t)
        : mixin(t.data())
    {
    }

    basic_tensor(const basic_tensor<R, basic_shape<0, Dim>, D, readwrite> &t)
        : mixin(t.data())
    {
    }

    operator const R &() const { return *this->data(); }
};

// rank > 0
template <typename R, typename S, typename D>
class basic_tensor<R, S, D, owner> : public basic_tensor_mixin<R, S, D, owner>
{
    using mixin = basic_tensor_mixin<R, S, D, owner>;
    using allocator = basic_allocator<R, D>;

    own_ptr<R, D> data_owner_;

    constexpr explicit basic_tensor(R *data, const S &shape)
        : mixin(data, shape), data_owner_(data)
    {
    }

  public:
    template <typename... Dims>
    constexpr explicit basic_tensor(Dims... d) : basic_tensor(S(d...))
    {
    }

    constexpr explicit basic_tensor(const S &shape)
        : basic_tensor(allocator()(shape.size()), shape)
    {
    }
};

template <typename R, typename S, typename D>
class basic_tensor<R, S, D, readwrite>
    : public basic_tensor_mixin<R, S, D, readwrite>
{
    using mixin = basic_tensor_mixin<R, S, D, readwrite>;

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

template <typename R, typename S, typename D>
class basic_tensor<R, S, D, readonly>
    : public basic_tensor_mixin<R, S, D, readonly>
{
    using mixin = basic_tensor_mixin<R, S, D, readonly>;

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

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readwrite> ref(const basic_tensor<R, S, D, owner> &t)
{
    return basic_tensor<R, S, D, readwrite>(t);
}

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readonly> view(const basic_tensor<R, S, D, owner> &t)
{
    return basic_tensor<R, S, D, readwrite>(t);
}

template <typename R, typename S, typename D>
basic_tensor<R, S, D, readonly> view(const basic_tensor<R, S, D, readwrite> &t)
{
    return basic_tensor<R, S, D, readonly>(t);
}

template <typename R, typename S, typename D, typename A> struct flattener {
    using S1 = typename S::template subshape_t<S::rank - 1>;
    using vector =
        basic_tensor<R, S1, D, typename basic_tensor_traits<R, A>::IterA>;

    vector operator()(const basic_tensor<R, S, D, A> &t) const
    {
        return vector(t.data(), t.shape().size());
    }
};

template <typename R, typename S, typename D, typename A>
typename flattener<R, S, D, A>::vector
flatten(const basic_tensor<R, S, D, A> &t)
{
    return flattener<R, S, D, A>()(t);
}
}  // namespace internal
}  // namespace ttl
