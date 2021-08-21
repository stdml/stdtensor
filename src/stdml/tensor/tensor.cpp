#include <cstring>
#include <sstream>

#include <stdml/bits/apply.hpp>
#include <stdml/bits/tensor/io.hpp>
#include <stdml/bits/tensor/tensor.hpp>
#include <ttl/bits/std_shape_debug.hpp>
#include <ttl/range>

namespace stdml
{
template <typename TT1>
Tensor Tensor::new_like(const BasicTensor<TT1> &x)
{
    return Tensor(x.value_type(), x.shape(), x.device());
}

template Tensor Tensor::new_like(const BasicTensor<raw_tensor> &);
template Tensor Tensor::new_like(const BasicTensor<raw_tensor_ref> &);
template Tensor Tensor::new_like(const BasicTensor<raw_tensor_view> &);

template <typename TT1>
Tensor Tensor::clone(const BasicTensor<TT1> &x)
{
    Tensor t(x.value_type(), x.shape(), x.device());
    if (x.device() == cpu) {
        std::memcpy(t.data(), x.data(), x.data_size());
    } else {
        throw std::runtime_error("only support clone cpu tensor.");
    }
    return t;
}

template Tensor Tensor::clone(const BasicTensor<raw_tensor> &);
template Tensor Tensor::clone(const BasicTensor<raw_tensor_ref> &);
template Tensor Tensor::clone(const BasicTensor<raw_tensor_view> &);

TensorView::TensorView(TT t, Device device) : P(std::move(t), device) {}

TensorView::TensorView(const Tensor &x) : P(TT(x.t_), x.device_) {}

TensorView::TensorView(const TensorRef &x) : P(TT(x.t_), x.device_) {}

TensorView TensorView::operator[](size_t i) const
{
    return TensorView(t_[i], device_);
}

TensorView TensorView::flatten() const
{
    return TensorView(t_.flatten(), device_);
}

TensorView TensorView::reshape(Shape s) const
{
    return TensorView(t_.reshape(s.get()), device_);
}

TensorView TensorView::chunk(size_t k) const
{
    return TensorView(t_.chunk(k), device_);
}

TensorView TensorView::slice(size_t i, size_t j) const
{
    return TensorView(t_.slice(i, j), device_);
}

TensorRef::TensorRef(TT t, Device device) : P(std::move(t), device) {}

TensorRef::TensorRef(const Tensor &x) : P(TT(x.t_), x.device_) {}

TensorView TensorRef::view() const { return TensorView(*this); }

TensorRef TensorRef::operator[](size_t i) const
{
    return TensorRef(t_[i], device_);
}

TensorRef TensorRef::flatten() const
{
    return TensorRef(t_.flatten(), device_);
}

TensorRef TensorRef::reshape(Shape s) const
{
    return TensorRef(t_.reshape(s.get()), device_);
}

TensorRef TensorRef::chunk(size_t k) const
{
    return TensorRef(t_.chunk(k), device_);
}

TensorRef TensorRef::slice(size_t i, size_t j) const
{
    return TensorRef(t_.slice(i, j), device_);
}

template <typename R>
struct show_scalar {
    void operator()(std::basic_ostream<char> &os, const R *ptr)
    {
        os << std::to_string(*ptr);
    }
};

struct bracket {
    char bra, ket;
};

template <typename R>
void show_tensor(std::basic_ostream<char> &os, const flat_shape &s,
                 const R *ptr, const bracket b)
{
    const dim_t limit = 10;
    if (s.rank() == 0) {
        show_scalar<R>()(os, ptr);
        return;
    }
    auto dims = s.dims();
    const auto ld = dims[0];
    dims.erase(dims.begin());
    const auto sub_shape = flat_shape(dims);
    auto sub_size = sub_shape.size();
    os << b.bra;
    for (auto i : ttl::range(ld)) {
        if (i > 0) { os << ", "; }
        if (i < limit) {
            show_tensor<R>(os, sub_shape, ptr + i * sub_size, b);
        } else {
            os << "...";
            break;
        }
    }
    os << b.ket;
}

struct show_tensor_t {
    std::basic_ostream<char> &os;
    bracket b;
    show_tensor_t(std::basic_ostream<char> &os, bracket b) : os(os), b(b) {}

    template <typename R>
    void operator()(const TensorView &x) const
    {
        show_tensor<R>(os, x.shape().get(), x.data<R>(), b);
    }
};

void show_tensor(std::basic_ostream<char> &os, const TensorView &x)
{
    if (x.device() != cpu) {
        throw std::invalid_argument(__func__ +
                                    std::string(" requires device=cpu"));
    }
    using E = Tensor::E;
    type_switch<E>()(to<E>(x.dtype()), show_tensor_t(os, {'[', ']'}), x);
}

void show_cpp(std::basic_ostream<char> &os, const TensorView &x)
{
    if (x.device() != cpu) {
        throw std::invalid_argument(__func__ +
                                    std::string(" requires device=cpu"));
    }
    using E = Tensor::E;
    type_switch<E>()(to<E>(x.dtype()), show_tensor_t(os, {'{', '}'}), x);
}

template <typename TT>
std::string gen_cpp(const TT &x)
{
    std::stringstream ss;
    show_cpp(ss, x);
    return ss.str();
}
}  // namespace stdml
