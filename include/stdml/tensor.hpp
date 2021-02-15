#pragma once
#include <list>

#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/raw_tensor.hpp>
#include <ttl/bits/std_encoding.hpp>
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/bits/type_encoder.hpp>
#include <ttl/tensor>

#include <stdml/dtype.hpp>
#include <stdml/tensor_config.hpp>

namespace ttl
{
namespace internal
{
// TODO: promote to ttl
template <typename A>
struct default_ref_type {
    using type = A;
};
template <>
struct default_ref_type<owner> {
    using type = readwrite;
};
}  // namespace internal
}  // namespace ttl

namespace stdml
{
template <typename T, typename C>
std::vector<T> cast_to(const C &xs)
{
    std::vector<T> ys(xs.size());
    std::copy(xs.begin(), xs.end(), ys.begin());
    return ys;
}

template <typename T, size_t r, typename C>
std::array<T, r> cast_to(const C &xs)
{
    if (xs.size() != r) { throw std::invalid_argument(__func__); }
    std::array<T, r> ys;
    std::copy(xs.begin(), xs.end(), ys.begin());
    return ys;
}

class Shape
{
    using S = flat_shape;

    const S s_;

  public:
    Shape() {}

    Shape(const std::list<long> &dims) : s_(cast_to<int64_t>(dims)) {}

    template <typename D, size_t r>
    Shape(const std::array<D, r> &dims) : s_(cast_to<int64_t>(dims))
    {
    }

    Shape(S s) : s_(std::move(s)) {}

    template <ttl::rank_t r>
    Shape(const ttl::shape<r> &s) : s_(s)
    {
    }

    // operator S() const { return s_; }

    const S &get() const { return s_; }

    size_t rank() const { return s_.rank(); }

    size_t size() const { return s_.size(); }
};

template <typename TT>
class BasicTensor
{
  protected:
    using S = flat_shape;
    using E = typename TT::encoder_type;
    using V = typename E::value_type;

    TT t_;

    BasicTensor(TT t) : t_(std::move(t)) {}

    using AA = typename ttl::internal::default_ref_type<
        typename TT::access_type>::type;

    template <typename R, typename A = AA>
    auto flatten() const
    {
        using vec =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<1>,
                                        ttl::internal::host_memory, A>;
        auto x = t_.template typed<R>();
        return vec(x.data(), x.size());
    }

  public:
    size_t rank() const { return t_.rank(); }

    size_t size() const { return t_.size(); }

    Shape shape() const { return t_.shape(); }

    DType dtype() const { return from<E>(t_.value_type()); }

    V value_type() const { return t_.value_type(); }

    size_t len() const
    {
        const auto &dims = t_.dims();
        if (dims.size() > 0) { return dims[0]; }
        return 0;
    }

    std::pair<V, S> meta() const { return {t_.value_type(), t_.shape()}; }

    template <size_t r>
    ttl::shape<r> ranked_shape() const
    {
        auto dims = cast_to<uint32_t, r>(t_.dims());
        return ttl::shape<r>(dims);
    }

    template <typename T>
    T _chunk(size_t k) const
    {
        auto dims = t_.dims();
        if (dims.size() <= 0) { throw std::invalid_argument(__func__); }
        auto ld = dims[0];
        dims.erase(dims.begin());
        dims.insert(dims.begin(), k);
        dims.insert(dims.begin(), ld / k);
        auto shape = flat_shape(dims);
        return raw_tensor_view(t_.data(), t_.value_type(), shape);
    }
};

class Tensor;

class TensorView : public BasicTensor<raw_tensor_view>
{
    using TT = raw_tensor_view;
    using P = BasicTensor<TT>;

  public:
    using P::E;
    using P::V;

    using P::flatten;

    TensorView(TT t);

    TensorView(const Tensor &);

    template <typename R>
    const R *data() const
    {
        return t_.data<R>();
    }

    TensorView operator[](size_t i) const
    {
        // TODO: check rank
        auto dims = t_.dims();
        dims.erase(dims.begin());
        auto sub_shape = flat_shape(dims);
        return raw_tensor_view((char *)t_.data() + i * sub_shape.size(),
                               t_.value_type(), sub_shape);
    }

    TensorView chunk(size_t k) const { return this->_chunk<TensorView>(k); }
};

class TensorRef : public BasicTensor<raw_tensor_ref>
{
    using TT = raw_tensor_ref;
    using P = BasicTensor<TT>;

  public:
    using P::E;
    using P::V;

    using P::flatten;

    TensorRef(TT t);

    TensorRef(const Tensor &);

    template <typename R>
    R *data() const
    {
        return t_.data<R>();
    }

    // TensorRef chunk(size_t k) const { return this->_chunk<TensorRef>(k); }
};

class Tensor : public BasicTensor<raw_tensor>
{
    //   protected:
    //     using S = flat_shape;
    //     using T = raw_tensor;
    //     T t_;
    using TT = raw_tensor;
    using P = BasicTensor<TT>;

    friend class TensorRef;
    friend class TensorView;

  public:
    using P::E;
    using P::V;

    using P::flatten;

    // using E = TT::encoder_type;
    // using V = E::value_type;

    template <typename TT1>
    static Tensor new_like(const BasicTensor<TT1> &x)
    {
        return Tensor(x.dtype(), x.shape());
    }

    Tensor(V v, const S &s) : P(TT(v, s)) {}

    Tensor(V v, const Shape &s) : P(TT(v, s.get())) {}

    Tensor(DType dt, const S &s) : P(TT(to<E>(dt), s)) {}

    Tensor(DType dt, const Shape &s) : P(TT(to<E>(dt), s.get())) {}

    Tensor(DType dt, const std::list<long> &dims) : Tensor(dt, Shape(dims)) {}

    bool match(const V v, const S &s) const
    {
        return v == t_.value_type() && s == t_.shape();
    }

    template <typename R>
    R *data() const
    {
        return t_.data<R>();
    }

    template <typename R>
    auto typed() const
    {
        return t_.typed<R>();
    }

    template <typename R, ttl::rank_t r, typename A = ttl::internal::readwrite>
    auto ranked() const
    {
        using tensor_ref =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<r>,
                                        ttl::internal::host_memory, A>;
        return tensor_ref(t_.data<R>(), ranked_shape<r>());
    }

    template <typename R, ttl::rank_t r>
    auto ranked_view() const
    {
        return ranked<R, r, ttl::internal::readonly>();
    }

    template <typename R>
    auto flatten_view() const
    {
        return flatten<R, ttl::internal::readonly>();
    }

    TensorRef ref() const { return TensorRef(*this); }

    TensorView view() const { return TensorView(*this); }

    TensorView operator[](size_t i) const
    {
        // TODO: check rank
        auto dims = t_.dims();
        dims.erase(dims.begin());
        auto sub_shape = flat_shape(dims);
        return raw_tensor_view((char *)t_.data() + i * sub_shape.size(),
                               t_.value_type(), sub_shape);
    }

    TensorView chunk(size_t k) const { return this->_chunk<TensorView>(k); }
};

std::string show(const Tensor &x);

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

template <typename F, ttl::rank_t r0, ttl::rank_t r1, ttl::rank_t r2>
struct apply<F, r0, r1, r2> {
    template <typename R>
    void operator()(const Tensor &z, const Tensor &x, const Tensor &y) const
    {
        F()(z.ranked<R, r0>(), x.ranked_view<R, r1>(), y.ranked_view<R, r2>());
    }
};
}  // namespace stdml
