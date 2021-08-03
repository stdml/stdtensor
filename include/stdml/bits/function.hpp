#pragma once
#include <vector>

#include <stdml/bits/apply.hpp>
#include <stdml/bits/tensor/tensor.hpp>

namespace stdml
{
using arity_t = uint8_t;

template <typename R, ttl::rank_t r>
struct type_rank {
    template <typename D, typename T>
    static auto on(const T &t)
    {
        return t.template typed<R, r, D>();
    }
};

template <typename R>
struct flat {
    template <typename D, typename T>
    static auto on(const T &t)
    {
        return t.template flatten<R, D>();
    }
};

template <typename R, ttl::rank_t r>
using tr = type_rank<R, r>;

template <typename D, typename TY, typename... TX>
class Apply
{
    using Y = std::vector<TensorRef>;
    using X = std::vector<TensorView>;

    static constexpr size_t arity = sizeof...(TX);

    template <typename F, std::size_t... I>
    static void apply(const F &f, const std::vector<TensorRef> &ys,
                      const std::vector<TensorView> &xs,
                      std::index_sequence<I...>)
    {
        using TTX = std::tuple<TX...>;
        f(TY::template on<D>(ys[0]),
          (std::tuple_element<I, TTX>::type::template on<D>(xs[I]))...);
    }

    template <typename P>
    static void check_arity(const P &p, size_t a)
    {
        static_assert(std::is_same<P, X>::value || std::is_same<P, Y>::value,
                      "");
        if (p.size() != a) { throw std::invalid_argument("invalid arity"); }
    }

  public:
    template <typename F>
    void operator()(const F &f, const Y &ys, const X &xs) const
    {
        check_arity(ys, 1);
        check_arity(xs, arity);
        apply(f, ys, xs, std::make_index_sequence<arity>());
    }
};

template <arity_t i, typename G, typename D, typename TY, typename... TX>
void apply_grad(Apply<D, TY, TX...>, const G &g,
                const std::vector<TensorRef> &ys,
                const std::vector<TensorView> &xs)
{
    using TTX = std::tuple<TX...>;
    using TXi = typename std::tuple_element<i, TTX>::type;
    Apply<D, TXi, TY, TY, TX...> grad;
    grad(g, ys, xs);
}

class Function
{
  protected:
    template <typename R, ttl::rank_t r>
    using tr = type_rank<R, r>;

    template <typename... T>
    using Apply = stdml::Apply<T...>;

    using Y = std::vector<TensorRef>;
    using X = std::vector<TensorView>;

    template <typename P>
    static void check_arity(const P &p, size_t a)
    {
        static_assert(std::is_same<P, X>::value || std::is_same<P, Y>::value,
                      "");
        if (p.size() != a) { throw std::invalid_argument("invalid arity"); }
    }

    using Kind = std::pair<DType, Shape>;

  public:
    static std::string call_info(const Y &ys, const X &xs);

    virtual ~Function() = default;

    virtual DType operator()(const std::vector<DType> &) const = 0;
    virtual Shape operator()(const std::vector<Shape> &) const = 0;

    Kind operator()(const std::vector<Kind> &) const;

    virtual void operator()(const Y &, const X &) const = 0;

    virtual void info() {}
};

template <typename F, typename TY, typename D = ttl::host_memory>
class Nullary : public Function
{
    using Function::check_arity;

  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        F f;
        Apply<D, TY>()(f, ys, xs);
    }
};

template <typename F, typename TY, typename TX, typename D = ttl::host_memory>
class Unary : public Function
{
  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        F f;
        Apply<D, TY, TX>()(f, ys, xs);
    }
};

template <size_t i, typename G, typename TY, typename TX,
          typename D = ttl::host_memory>
class dUnary : public Function
{
  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        static_assert(i == 0, "");
        G g;
        Apply<D, TX, TY, TY, TX>()(g, ys, xs);
    }
};

template <typename F, typename TY, typename TX0, typename TX1,
          typename D = ttl::host_memory>
class Binary : public Function
{
  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        F f;
        Apply<D, TY, TX0, TX1>()(f, ys, xs);
    }
};

template <size_t i, typename F, typename G, typename TY, typename TX0,
          typename TX1, typename D = ttl::host_memory>
class dBinary : public Function
{
  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        static_assert(i < 2, "");
        using TTX = std::tuple<TX0, TX1>;
        using TXi = typename std::tuple_element<i, TTX>::type;
        F f;
        G g(f);
        Apply<D, TXi, TY, TY, TX0, TX1>()(g, ys, xs);
    }
};

template <typename F, typename D, ttl::rank_t q, ttl::rank_t p>
class GenericUnary : public Function
{
    using Function ::check_arity;

  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        check_arity(ys, 1);
        check_arity(xs, 1);
        const auto &x = xs[0];
        const auto &y = ys[0];
        // TODO: check dtype and shape
        using E = Tensor::E;
        type_switch_float<E>()(y.value_type(), apply<F, D, q, p>(), y, x);
    }

    Tensor __call__(const TensorView &x) const
    {
        auto shape = F()(x.shape().ranked<p, uint32_t>());
        Tensor y = Tensor(x.value_type(), shape);
        operator()({y.ref()}, {x});
        return y;
    }
};

template <typename F, typename D = ttl::host_memory>
class GenericBinaryPointwise : public Function
{
    using Function::check_arity;

  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        check_arity(ys, 1);
        check_arity(xs, 2);
        const auto &x0 = xs[0];
        const auto &x1 = xs[1];
        const auto &y = ys[0];
        // TODO: check dtype and shape
        using E = Tensor::E;
        type_switch<E>()(y.value_type(), apply<F, D>(), y, x0, x1);
    }
};

template <typename F, ttl::rank_t r, ttl::rank_t p, ttl::rank_t q,
          typename D = ttl::host_memory>
class GenericBinary : public Function
{
    using Function ::check_arity;

  public:
    void operator()(const std::vector<TensorRef> &ys,
                    const std::vector<TensorView> &xs) const override
    {
        check_arity(ys, 1);
        check_arity(xs, 2);
        const auto &x0 = xs[0];
        const auto &x1 = xs[1];
        const auto &y = ys[0];
        // TODO: check dtype and shape
        using E = Tensor::E;
        type_switch<E>()(y.value_type(), apply<F, D, r, p, q>(), y, x0, x1);
    }
};
}  // namespace stdml
