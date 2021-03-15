#pragma once
#include <list>

#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/raw_tensor.hpp>
#include <ttl/bits/std_encoding.hpp>
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/bits/type_encoder.hpp>
#include <ttl/device>
#include <ttl/tensor>

#include <stdml/bits/device.hpp>
#include <stdml/bits/dtype.hpp>
#include <stdml/bits/shape.hpp>
#include <stdml/bits/tensor_config.hpp>

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
class TensorMeta
{
    // dtype
    // shape
    // device
    // is_dense ?
  public:
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

    template <typename R, ttl::rank_t r, typename A = AA>
    auto ranked() const
    {
        using tsr =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<r>,
                                        ttl::internal::host_memory, A>;
        return tsr(t_.template data<R>(), ranked_shape<r>());
    }

  public:
    size_t rank() const { return t_.rank(); }

    size_t size() const { return t_.size(); }

    size_t data_size() const { return t_.data_size(); }

    Shape shape() const { return t_.shape(); }

    DType dtype() const { return from<E>(t_.value_type()); }

    V value_type() const { return t_.value_type(); }

    size_t len() const
    {
        const auto &dims = t_.dims();
        if (dims.size() > 0) { return dims[0]; }
        return 0;
    }

    const auto &dims() const { return t_.dims(); }

    std::pair<V, S> meta() const { return {t_.value_type(), t_.shape()}; }

    template <size_t r>
    ttl::shape<r> ranked_shape() const
    {
        auto dims = cast_to<uint32_t, r>(t_.dims());
        return ttl::shape<r>(dims);
    }

    // pointer accessors

    auto data() const { return t_.data(); }

    template <typename R>
    auto data() const
    {
        return t_.template data<R>();
    }

    template <typename R>
    auto data_end() const
    {
        return t_.template data<R>() + t_.size();
    }

    template <typename R>
    auto typed() const
    {
        using tsr = ttl::internal::basic_tensor<
            R, ttl::internal::basic_flat_shape<uint32_t>,
            ttl::internal::host_memory, AA>;
        auto dims = cast_to<uint32_t>(t_.dims());
        return tsr(t_.template data<R>(), std::move(dims));
    }

    template <typename R, ttl::rank_t r>
    auto typed() const
    {
        using tsr =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<r>,
                                        ttl::internal::host_memory, AA>;
        return tsr(t_.template data<R>(), ranked_shape<r>());
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

    template <typename T>
    T _slice(size_t i, size_t j) const
    {
        return t_.slice(i, j);
    }
};

class Tensor;
class TensorRef;

class TensorView : public BasicTensor<raw_tensor_view>
{
    using TT = raw_tensor_view;
    using P = BasicTensor<TT>;

  public:
    using P::E;
    using P::V;

    using P::flatten;
    using P::ranked;

    TensorView(TT t);

    template <typename R, ttl::rank_t r>
    TensorView(ttl::tensor_view<R, r> x) : TensorView(raw_tensor_view(x))
    {
    }

    TensorView(const Tensor &);

    TensorView(const TensorRef &);

    TensorView operator[](size_t i) const
    {
        // TODO: check rank
        auto dims = t_.dims();
        dims.erase(dims.begin());
        auto sub_shape = flat_shape(dims);
        char *offset =
            (char *)t_.data() + i * sub_shape.size() * E::size(t_.value_type());
        return raw_tensor_view(offset, t_.value_type(), sub_shape);
    }

    TensorView chunk(size_t k) const { return this->_chunk<TensorView>(k); }

    TensorView slice(size_t i, size_t j)
    {
        return this->_slice<TensorView>(i, j);
    }
};

class TensorRef : public BasicTensor<raw_tensor_ref>
{
    using TT = raw_tensor_ref;
    using P = BasicTensor<TT>;

    friend class TensorView;

  public:
    using P::E;
    using P::V;

    using P::flatten;
    using P::ranked;

    TensorRef(TT t);

    template <typename R, ttl::rank_t r>
    TensorRef(ttl::tensor_ref<R, r> x) : TensorRef(raw_tensor_ref(x))
    {
    }

    TensorRef(const Tensor &);

    // TensorRef chunk(size_t k) const { return this->_chunk<TensorRef>(k); }

    TensorView view() const { return TensorView(*this); }

    TensorRef slice(size_t i, size_t j)
    {
        return this->_slice<TensorRef>(i, j);
    }
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
    using P::ranked;

    // using E = TT::encoder_type;
    // using V = E::value_type;

    template <typename TT1>
    static Tensor new_like(const BasicTensor<TT1> &x)
    {
        return Tensor(x.dtype(), x.shape());
    }

    // Tensor(V v) : P(TT(v, S())) {}

    template <typename R, ttl::rank_t r>
    Tensor(ttl::tensor<R, r> x) : P(TT(std::move(x)))
    {
    }

    Tensor(V v, const S &s) : P(TT(v, s)) {}

    Tensor(V v, const Shape &s) : P(TT(v, s.get())) {}

    template <typename D>
    Tensor(DType dt, std::initializer_list<D> dims)
        : Tensor(dt, std::vector<D>(std::move(dims)))
    {
    }

    template <typename D>
    Tensor(DType dt, std::vector<D> dims) : Tensor(dt, Shape(std::move(dims)))
    {
    }

    Tensor(DType dt) : P(TT(to<E>(dt), S())) {}

    template <ttl::rank_t r>
    Tensor(DType dt, const ttl::shape<r> &s) : Tensor(dt, S(s))
    {
    }

    Tensor(DType dt, const S &s) : P(TT(to<E>(dt), s)) {}

    Tensor(DType dt, const Shape &s) : P(TT(to<E>(dt), s.get())) {}

    Tensor(DType dt, const std::list<long> &dims) : Tensor(dt, Shape(dims)) {}

    bool match(const V v, const S &s) const
    {
        return v == t_.value_type() && s == t_.shape();
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
        char *offset =
            (char *)t_.data() + i * sub_shape.size() * E::size(t_.value_type());
        return raw_tensor_view(offset, t_.value_type(), sub_shape);
    }

    TensorView chunk(size_t k) const { return this->_chunk<TensorView>(k); }

    TensorRef slice(size_t i, size_t j)
    {
        return this->_slice<TensorRef>(i, j);
    }
};
}  // namespace stdml
