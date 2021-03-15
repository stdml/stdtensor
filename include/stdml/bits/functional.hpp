#pragma once
#include <stdml/bits/tensor.hpp>

namespace stdml
{
template <typename E>
struct type_switch {
    template <typename F, typename... Args>
    decltype(auto) operator()(typename E::value_type vt, const F &f,
                              const Args &... args) const
    {
#define CASE(T)                                                                \
    case E::template value<T>():                                               \
        return f.template operator()<T>(args...);

        switch (vt) {
            // TODO: list of types should depend on E
            CASE(uint8_t);
            CASE(int8_t);
            CASE(int16_t);
            CASE(int32_t);
            CASE(float);
            CASE(double);
        default:
            throw std::invalid_argument("unsupported value type: " +
                                        std::to_string(vt));
        }
#undef CASE
    }
};

template <typename E>
struct type_switch_float {
    template <typename F, typename... Args>
    decltype(auto) operator()(typename E::value_type vt, const F &f,
                              const Args &... args) const
    {
#define CASE(T)                                                                \
    case E::template value<T>():                                               \
        return f.template operator()<T>(args...);

        switch (vt) {
            CASE(float);
            CASE(double);
        default:
            throw std::invalid_argument("unsupported value type: " +
                                        std::to_string(vt));
        }
#undef CASE
    }
};

template <typename F, ttl::rank_t... ranks>
struct apply;

template <typename F>
struct apply<F> {
    template <typename R>
    void operator()(const Tensor &x) const
    {
        F()(x.flatten<R>());
    }

    template <typename R>
    void operator()(const Tensor &y, const Tensor &x) const
    {
        F()(y.flatten<R>(), x.flatten_view<R>());
    }

    template <typename R>
    void operator()(const Tensor &z, const Tensor &x, const Tensor &y) const
    {
        F()(z.flatten<R>(), x.flatten_view<R>(), y.flatten_view<R>());
    }

    template <typename R>
    void operator()(const TensorRef &z,  //
                    const TensorView &x, const TensorView &y) const
    {
        F()(z.flatten<R>(), x.flatten<R>(), y.flatten<R>());
    }
};

template <typename F, ttl::rank_t r0, ttl::rank_t r1>
struct apply<F, r0, r1> {

    template <typename R>
    void operator()(const TensorRef &y, const TensorView &x) const
    {
        F()(y.ranked<R, r0>(), x.ranked<R, r1>());
    }
};

template <typename F, ttl::rank_t r0, ttl::rank_t r1, ttl::rank_t r2>
struct apply<F, r0, r1, r2> {
    template <typename R>
    void operator()(const Tensor &z, const Tensor &x, const Tensor &y) const
    {
        F()(z.ranked<R, r0>(), x.ranked_view<R, r1>(), y.ranked_view<R, r2>());
    }

    template <typename R>
    void operator()(const TensorRef &z,  //
                    const TensorView &x, const TensorView &y) const
    {
        F()(z.ranked<R, r0>(), x.ranked<R, r1>(), y.ranked<R, r2>());
    }
};
}  // namespace stdml
