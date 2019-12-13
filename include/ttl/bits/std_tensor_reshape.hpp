#pragma once
#include <ttl/bits/std_tensor.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
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

template <typename R, typename S, typename D, typename A>
struct flattener {
    using S1 = typename S::template subshape_t<S::rank - 1>;
    using vector =
        basic_tensor<R, S1, D, typename basic_tensor_traits<R, A, D>::Access>;

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
