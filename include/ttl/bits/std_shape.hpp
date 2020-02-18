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
template <typename N, typename Iterator>
N product(Iterator begin, Iterator end)
{
    return std::accumulate(begin, end, static_cast<N>(1), std::multiplies<N>());
};

template <size_t off, typename T, size_t r, size_t... Is>
constexpr std::array<T, r - 1> shift_idx(const std::array<T, r> &a,
                                         std::index_sequence<Is...>)
{
    return std::array<T, r - 1>({std::get<Is + off>(a)...});
}

using rank_t = uint8_t;

template <rank_t r, typename Dim = uint32_t>
class basic_shape
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

    template <typename... I>
    dim_t offset(I... args) const
    {
        static_assert(sizeof...(I) == r, "invalid number of indexes");

        // TODO: expand the expression
        const std::array<dim_t, r> offs({static_cast<dim_t>(args)...});
        dim_t off = 0;
        for (rank_t i = 0; i < r; ++i) { off = off * dims_[i] + offs[i]; }
        return off;
    }

    std::array<dim_t, r> expand(dim_t off) const
    {
        std::array<dim_t, r> coords;
        for (rank_t i = r; i > 0; --i) {
            coords[i - 1] = off % dims_[i - 1];
            off /= dims_[i - 1];
        }
        return coords;
    }

    template <rank_t p>
    dim_t coord(dim_t off) const
    {
        static_assert(p < rank, "invalid coordinate");
        for (rank_t i = p + 1; i < rank; ++i) { off /= dims_[i]; }
        off %= std::get<p>(dims_);
        return off;
    }

    dim_t size() const { return product<dim_t>(dims_.begin(), dims_.end()); }

    dim_t subspace_size() const
    {
        return product<dim_t>(dims_.begin() + 1, dims_.end());
    }

    template <rank_t corank = 1>
    using subshape_t = basic_shape<r - corank, dim_t>;

    template <rank_t corank = 1>
    basic_shape<r - corank, dim_t> subshape() const
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
}  // namespace internal
}  // namespace ttl
