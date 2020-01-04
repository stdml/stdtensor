#pragma once
#include <ttl/bits/std_shape_algo.hpp>
#include <ttl/bits/std_tensor.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D, typename A>
struct flattener {
    using S1 = basic_shape<1, typename S::dimension_type>;
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

template <typename R, typename S, typename D, typename A>
struct chunker {
    using S1 = typename super_shape<S>::type;
    using A1 = typename basic_tensor_traits<R, A, D>::Access;
    using T1 = basic_tensor<R, S1, D, A1>;
    using T = basic_tensor<R, S, D, A>;
    using dim_t = typename T::shape_type::dimension_type;

    static constexpr rank_t r = S::rank;

    T1 operator()(const T &t, const dim_t &k, dim_t &dropped) const
    {
        static_assert(r > 0, "rank > 0 is requied");
        const dim_t l = std::get<0>(t.shape().dims());
        const dim_t n = l / k;
        dropped = (l - n * k);
        const std::array<dim_t, r - 1> sub_dims = t.shape().subshape().dims();
        const auto dim_tup = std::tuple_cat(std::make_tuple(n, k), sub_dims);
        return T1(t.data(), tup2arr<dim_t>(dim_tup));
    }
};

template <typename R, typename S, typename D, typename A>
typename chunker<R, S, D, A>::T1 chunk(const basic_tensor<R, S, D, A> &t, int k)
{
    typename chunker<R, S, D, A>::dim_t dropped;
    return chunker<R, S, D, A>()(t, k, dropped);
    // FIXME: support throw if dropped > 0
}
}  // namespace internal
}  // namespace ttl
