#pragma once
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
template <typename I, I...>
struct int_seq_sum;

template <typename I>
struct int_seq_sum<I> {
    static constexpr I value = 0;
};

template <typename I, I i0, I... is>
struct int_seq_sum<I, i0, is...> {
    static constexpr I value = i0 + int_seq_sum<I, is...>::value;
};

template <typename T, size_t p, size_t q, size_t... Is, size_t... Js>
constexpr std::array<T, p + q>
merge_indexed(const std::array<T, p> &a, std::index_sequence<Is...>,
              const std::array<T, q> &b, std::index_sequence<Js...>)
{
    return std::array<T, p + q>({std::get<Is>(a)..., std::get<Js>(b)...});
}

template <typename... D>
constexpr basic_shape<sizeof...(D)> make_shape(const D... d)
{
    return basic_shape<sizeof...(D)>(d...);
}

template <rank_t p, rank_t q, typename dim_t>
constexpr basic_shape<p + q, dim_t> join_shape(const basic_shape<p, dim_t> &s,
                                               const basic_shape<q, dim_t> &t)
{
    return basic_shape<p + q, dim_t>(
        merge_indexed(s.dims(), std::make_index_sequence<p>(),  //
                      t.dims(), std::make_index_sequence<q>()));
}

template <rank_t r, typename dim_t, typename N = dim_t>
basic_shape<r + 1, dim_t> batch(const N n, const basic_shape<r, dim_t> &s)
{
    return join_shape(basic_shape<1, dim_t>(n), s);
}

template <rank_t r, typename dim_t, typename N = dim_t>
basic_shape<r + 1, dim_t> vectorize(const basic_shape<r, dim_t> &s,
                                    const N d = 1)
{
    return join_shape(s, basic_shape<1, dim_t>(d));
}

template <rank_t... rs>
class flatten_shape;

template <>
class flatten_shape<>
{
  public:
    template <rank_t r, typename dim_t>
    basic_shape<1, dim_t> operator()(const basic_shape<r, dim_t> &shape) const
    {
        return basic_shape<1, dim_t>(shape.size());
    }
};

template <rank_t... rs>
class flatten_shape
{
    static constexpr rank_t in_rank = int_seq_sum<int, rs...>::value;
    static constexpr rank_t out_rank = sizeof...(rs);

  public:
    template <typename dim_t>
    basic_shape<out_rank, dim_t>
    operator()(const basic_shape<in_rank, dim_t> &shape) const
    {
        constexpr std::array<rank_t, out_rank> ranks({rs...});
        std::array<dim_t, out_rank> dims;
        rank_t j = 0;
        for (rank_t i = 0; i < out_rank; ++i) {
            dims[i] = product<dim_t>(shape.dims().begin() + j,
                                     shape.dims().begin() + j + ranks[i]);
            j += ranks[i];
        }
        return basic_shape<out_rank, dim_t>(dims);
    }
};

template <rank_t r, typename dim_t>
basic_shape<1, dim_t> flatten(const basic_shape<r, dim_t> &s)
{
    return flatten_shape<r>()(s);
}

template <typename S>
struct super_shape;

template <rank_t r, typename D>
struct super_shape<basic_shape<r, D>> {
    using type = basic_shape<r + 1, D>;
};

template <typename T, typename Tuple, size_t... I>
std::array<T, std::tuple_size<Tuple>::value> tup2arr(const Tuple &t,
                                                     std::index_sequence<I...>)
{
    using Array = std::array<T, std::tuple_size<Tuple>::value>;
    return Array({static_cast<T>(std::get<I>(t))...});
}

template <typename T, typename Tuple>
std::array<T, std::tuple_size<Tuple>::value> tup2arr(const Tuple &t)
{
    return tup2arr<T>(
        t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}
}  // namespace internal
}  // namespace ttl
