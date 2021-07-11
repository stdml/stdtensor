#pragma once
#include <list>

#include <ttl/bits/flat_shape.hpp>
#include <ttl/bits/raw_tensor.hpp>
#include <ttl/bits/std_encoding.hpp>
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/bits/type_encoder.hpp>
#include <ttl/device>
#include <ttl/tensor>

#include <stdml/bits/tensor/allocator.hpp>
#include <stdml/bits/tensor/device.hpp>
#include <stdml/bits/tensor/dtype.hpp>
#include <stdml/bits/tensor/shape.hpp>
#include <stdml/bits/tensor/tensor_config.hpp>

namespace stdml
{
class TensorMeta
{
    // dtype
    // shape
    // device
    // is_dense ?

  protected:
    using S = flat_shape;
    using E = typename stdml::raw_tensor::encoder_type;
    using V = typename E::value_type;

    V value_type_;
    S shape_;
    Device device_;

  public:
    TensorMeta(V value_type, S shape, Device device = cpu)
        : value_type_(value_type), shape_(std::move(shape)), device_(device)
    {
    }
};

template <typename TT>
class BasicTensor
{
  protected:
    using E = typename TT::encoder_type;
    using V = typename E::value_type;

    const Device device_;
    TT t_;

    explicit BasicTensor(TT t, Device device)
        : device_(device), t_(std::move(t))
    {
    }

    using AA = typename ttl::internal::basic_access_traits<
        typename TT::access_type>::type;

  public:
    size_t rank() const { return t_.rank(); }

    size_t size() const { return t_.size(); }

    size_t data_size() const { return t_.data_size(); }

    Shape shape() const { return t_.shape(); }

    DType dtype() const { return from<E>(t_.value_type()); }

    V value_type() const { return t_.value_type(); }

    Device device() const { return device_; }

    size_t len() const
    {
        const auto &dims = t_.dims();
        if (dims.size() > 0) { return dims[0]; }
        return 0;
    }

    const auto &dims() const { return t_.dims(); }

    // std::pair<V, S> meta() const { return {t_.value_type(), t_.shape()}; }

    // pointer accessors

    auto data() const { return t_.data(); }

    template <Device device>
    auto data() const
    {
        if (device != device_) {
            throw ttl::internal::invalid_device_reification();
        }
        return t_.data();
    }

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

    template <typename R, typename... Dim>
    auto at(Dim... i) const
    {
        using D = ttl::internal::host_memory;
        if (device_type<D>::value != device_) {
            throw ttl::internal::invalid_device_reification();
        }
        constexpr ttl::rank_t r = sizeof...(Dim);
        auto t = typed<R, r, D>();
        using dim_t = typename decltype(t)::shape_type::dimension_type;
        return t.at(static_cast<dim_t>(i)...);
    }

    template <typename R, typename D = ttl::internal::host_memory>
    auto flatten() const
    {
        if (device_type<D>::value != device_) {
            throw ttl::internal::invalid_device_reification();
        }
        using vec =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<1>, D,
                                        AA>;
        auto x = t_.template typed<R>();
        return vec(x.data(), x.size());
    }

    template <typename R, typename D = ttl::internal::host_memory>
    auto typed() const
    {
        if (device_type<D>::value != device_) {
            throw ttl::internal::invalid_device_reification();
        }
        using tsr =
            ttl::internal::basic_tensor<R, ttl::internal::basic_flat_shape<>, D,
                                        AA>;
        auto dims = cast_to<uint32_t>(t_.dims());
        return tsr(t_.template data<R>(), std::move(dims));
    }

    template <typename R, ttl::rank_t r,
              typename D = ttl::internal::host_memory>
    auto typed() const
    {
        if (device_type<D>::value != device_) {
            throw ttl::internal::invalid_device_reification();
        }
        using tsr =
            ttl::internal::basic_tensor<R, ttl::internal::basic_shape<r>, D,
                                        AA>;
        auto dims = cast_to<uint32_t, r>(t_.dims());
        return tsr(t_.template data<R>(), ttl::internal::basic_shape<r>(dims));
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
    using P::flatten;
    using P::V;

    TensorView(TT t, Device device = cpu);

    template <typename R, ttl::rank_t r, typename D>
    TensorView(ttl::tensor_view<R, r, D> x)
        : TensorView(raw_tensor_view(x), device_type<D>::value)
    {
    }

    TensorView(const Tensor &);

    TensorView(const TensorRef &);

    TensorView operator[](size_t i) const { return TensorView(t_[i], device_); }

    TensorView flatten() const { return TensorView(t_.flatten(), device_); }

    TensorView chunk(size_t k) const
    {
        return TensorView(t_.chunk(k), device_);
    }

    TensorView slice(size_t i, size_t j) const
    {
        return TensorView(t_.slice(i, j), device_);
    }
};

class TensorRef : public BasicTensor<raw_tensor_ref>
{
    using TT = raw_tensor_ref;
    using P = BasicTensor<TT>;

    friend class TensorView;

  public:
    using P::E;
    using P::flatten;
    using P::V;

    TensorRef(TT t, Device device = cpu);

    template <typename R, ttl::rank_t r, typename D>
    TensorRef(ttl::tensor_ref<R, r, D> x)
        : TensorRef(raw_tensor_ref(x), device_type<D>::value)
    {
    }

    TensorRef(const Tensor &);

    TensorView view() const { return TensorView(*this); }

    TensorRef operator[](size_t i) const { return TensorRef(t_[i], device_); }

    TensorRef flatten() const { return TensorRef(t_.flatten(), device_); }

    TensorRef chunk(size_t k) const { return TensorRef(t_.chunk(k), device_); }

    TensorRef slice(size_t i, size_t j) const
    {
        return TensorRef(t_.slice(i, j), device_);
    }
};

class Tensor : public BasicTensor<raw_tensor_ref>
{
    using TT = raw_tensor_ref;
    using P = BasicTensor<TT>;

    friend class TensorRef;
    friend class TensorView;

    Buffer buffer_;

    Tensor(void *data, Device device, V v, flat_shape s)
        : P(TT(data, v, s), device), buffer_(data, device)
    {
    }

  public:
    using P::E;
    using P::flatten;
    using P::V;

    template <typename TT1>
    static Tensor new_like(const BasicTensor<TT1> &x)
    {
        return Tensor(x.value_type(), x.shape(), x.device());
    }

    Tensor(V v, Device device = cpu) : Tensor(v, flat_shape(), device) {}

    template <typename R, ttl::rank_t r>
    Tensor(ttl::tensor<R, r> x)
        : Tensor(x.data_.release(), cpu, E::value<R>(), flat_shape(x.shape()))
    {
    }

    Tensor(V v, const flat_shape &s, Device device = cpu)
        : Tensor(GA::alloc(device, s.size() * E::size(v)), device, v, s)
    {
    }

    Tensor(V v, const Shape &s, Device device = cpu)
        : Tensor(v, s.get(), device)
    {
    }

    template <typename D>
    Tensor(DType dt, std::initializer_list<D> dims)
        : Tensor(dt, std::vector<D>(std::move(dims)))
    {
    }

    template <typename D>
    Tensor(DType dt, std::vector<D> dims) : Tensor(dt, Shape(std::move(dims)))
    {
    }

    Tensor(DType dt, Device device = cpu)
        : Tensor(to<E>(dt), flat_shape(), device)
    {
    }

    template <ttl::rank_t r>
    Tensor(DType dt, const ttl::shape<r> &s, Device device = cpu)
        : Tensor(dt, flat_shape(s), device)
    {
    }

    Tensor(DType dt, const flat_shape &s, Device device = cpu)
        : Tensor(to<E>(dt), s, device)
    {
    }

    Tensor(DType dt, const Shape &s, Device device = cpu)
        : Tensor(to<E>(dt), s.get(), device)
    {
    }

    Tensor(DType dt, const std::list<long> &dims, Device device = cpu)
        : Tensor(dt, Shape(dims), device)
    {
    }

    TensorRef ref() const { return TensorRef(*this); }

    TensorView view() const { return TensorView(*this); }

    TensorRef flatten() const { return TensorRef(t_.flatten(), device_); }

    TensorRef operator[](size_t i) const { return TensorRef(t_[i], device_); }

    TensorRef chunk(size_t k) const { return TensorRef(t_.chunk(k), device_); }

    TensorRef slice(size_t i, size_t j) const
    {
        return TensorRef(t_.slice(i, j), device_);
    }
};
}  // namespace stdml
