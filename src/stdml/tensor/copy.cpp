#include <stdml/bits/tensor/allocator.hpp>
#include <stdml/bits/tensor/io.hpp>
#include <ttl/consistent_variable>

namespace stdml
{
void copy(const TensorRef &y, const TensorView &x)
{
    // ttl::consistent_variable<size_t> n(x.size());
    // n = y.size();
    // using GA = generic_allocator;
    // if (x.device() == cpu && y.device() == cpu) {
    //     GA::_h2h(y.data(), x.data(), n);
    //     return;
    // }
    // if (x.device() == cpu && y.device() == cpu) {
    //     GA::_h2h(y.data(), x.data(), n);
    //     return;
    // }
    // if (x.device() == cpu && y.device() == cpu) {
    //     GA::_h2h(y.data(), x.data(), n);
    //     return;
    // }
    // if (x.device() == cpu && y.device() == cpu) {
    //     GA::_h2h(y.data(), x.data(), n);
    //     return;
    // }
}
}  // namespace stdml
