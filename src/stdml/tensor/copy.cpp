#include <cstring>

#include <stdml/bits/tensor/allocator.hpp>
#include <stdml/bits/tensor/cudart.hpp>
#include <stdml/bits/tensor/io.hpp>
#include <ttl/consistent_variable>

namespace stdml
{
namespace ops
{
void copy::operator()(const TensorRef &y, const TensorView &x) const
{
    const size_t n = x.data_size();
    if (n != y.data_size()) {
        throw std::invalid_argument("copy: inconsistent size");
    }
    if (x.device() == cpu && y.device() == cpu) {
        std::memcpy(y.data(), x.data(), n);
        return;
    }
    auto &l = libcudart::get();
    if (x.device() == cpu && y.device() == cuda) {
        l.from_host(y.data(), x.data(), n);
    } else if (x.device() == cuda && y.device() == cpu) {
        l.to_host(y.data(), x.data(), n);
    } else if (x.device() == cuda && y.device() == cuda) {
        l.d2d(y.data(), x.data(), n);
    } else {
        throw std::invalid_argument("stdml::ops::copy");
    }
}
}  // namespace ops

void copy(const TensorRef &y, const TensorView &x) { ops::copy()(y, x); }
}  // namespace stdml
