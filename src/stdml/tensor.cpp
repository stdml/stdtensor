#include <sstream>

#include <stdml/tensor.hpp>
#include <ttl/bits/std_shape_debug.hpp>
#include <ttl/range>

namespace stdml
{
TensorView::TensorView(TT t) : P(std::move(t)) {}

TensorView::TensorView(const Tensor &x) : P(TT(x.t_)) {}

TensorRef::TensorRef(const Tensor &x) : P(TT(x.t_)) {}

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
    using E = Tensor::E;
    type_switch<E>()(to<E>(x.dtype()), show_tensor_t(os, {'[', ']'}), x);
}

void show_cpp(std::basic_ostream<char> &os, const TensorView &x)
{
    using E = Tensor::E;
    type_switch<E>()(to<E>(x.dtype()), show_tensor_t(os, {'{', '}'}), x);
}

template <typename TT>
std::string show(const TT &x)
{
    std::stringstream ss;
    ss << tn(x.dtype()) << ttl::internal::to_string(x.shape().get());
    show_tensor(ss, x);
    return ss.str();
}

template <typename TT>
std::string gen_cpp(const TT &x)
{
    std::stringstream ss;
    show_cpp(ss, x);
    return ss.str();
}
}  // namespace stdml
