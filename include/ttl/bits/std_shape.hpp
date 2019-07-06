#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

namespace ttl
{
namespace internal
{
template <size_t off, typename T, size_t r, size_t... Is>
constexpr std::array<T, r - 1> shift_idx(const std::array<T, r> &a,
                                         std::index_sequence<Is...>)
{
    return std::array<T, r - 1>({std::get<Is + off>(a)...});
}

template <typename T, size_t p, size_t q, size_t... Is, size_t... Js>
constexpr std::array<T, p + q>
merge_indexed(const std::array<T, p> &a, std::index_sequence<Is...>,
              const std::array<T, q> &b, std::index_sequence<Js...>)
{
    return std::array<T, p + q>({std::get<Is>(a)..., std::get<Js>(b)...});
}

using rank_t = uint8_t;

template <rank_t r, typename Dim = uint32_t> class basic_shape
{
    using dim_t = Dim;

    const std::array<dim_t, r> dims_;

  public:
    using dimension_type = Dim;

    static constexpr rank_t rank = r;

    constexpr explicit basic_shape(const std::array<dim_t, r> &dims)
        : dims_(dims)
    {
    }

    template <typename... D>
    constexpr explicit basic_shape(D... d) : dims_({static_cast<dim_t>(d)...})
    {
        static_assert(sizeof...(D) == r, "invalid number of dims");
    }

    template <typename... I> dim_t offset(I... args) const
    {
        static_assert(sizeof...(I) == r, "invalid number of indexes");

        // TODO: expand the expression
        const std::array<dim_t, r> offs({static_cast<dim_t>(args)...});
        dim_t off = 0;
        for (rank_t i = 0; i < r; ++i) { off = off * dims_[i] + offs[i]; }
        return off;
    }

    dim_t size() const
    {
        return std::accumulate(dims_.begin(), dims_.end(),
                               static_cast<dim_t>(1), std::multiplies<dim_t>());
    }

    dim_t subspace_size() const
    {
        return std::accumulate(dims_.begin() + 1, dims_.end(),
                               static_cast<dim_t>(1), std::multiplies<dim_t>());
    }

    template <rank_t corank = 1>
    using subshape_t = basic_shape<r - corank, dim_t>;

    template <rank_t corank = 1> basic_shape<r - corank, dim_t> subshape() const
    {
        static_assert(0 <= corank && corank <= r, "invalid corank");
        constexpr rank_t s = r - corank;
        return basic_shape<s, dim_t>(
            shift_idx<corank>(dims_, std::make_index_sequence<s>()));
    }

    bool operator==(const basic_shape &s) const
    {
        return std::equal(dims_.begin(), dims_.end(), s.dims().begin());
    }

    bool operator!=(const basic_shape &s) const { return !operator==(s); }

    const std::array<dim_t, r> &dims() const { return dims_; }
};

template <typename... D> basic_shape<sizeof...(D)> make_shape(const D... d)
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

template <rank_t r, typename dim_t>
basic_shape<r + 1, dim_t>
batch(typename basic_shape<r, dim_t>::dimension_type n,
      const basic_shape<r, dim_t> &s)
{
    return internal::join_shape(basic_shape<1, dim_t>(n), s);
}

}  // namespace internal
}  // namespace ttl
