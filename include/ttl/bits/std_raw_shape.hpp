#pragma once
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
template <typename D_out, typename D_in, rank_t r>
std::vector<D_out> arr2vec(const std::array<D_in, r> &a)
{
    std::vector<D_out> v(r);
    std::copy(a.begin(), a.end(), v.begin());
    return v;
}

template <typename Dim = uint32_t> class basic_raw_shape
{
    using dim_t = Dim;

  public:
    using dimension_type = Dim;

    template <typename... D>
    explicit basic_raw_shape(D... d) : dims({static_cast<dim_t>(d)...})
    {
        static_assert(sizeof...(D) <= 0xff, "too many dimensions");
    }

    explicit basic_raw_shape(const std::vector<dim_t> &dims) : dims(dims) {}

    template <rank_t r, typename D>
    explicit basic_raw_shape(const basic_shape<r, D> &shape)
        : dims(std::move(arr2vec<Dim, D, r>(shape.dims())))
    {
    }

    rank_t rank() const { return dims.size(); }

    dim_t size() const
    {
        return std::accumulate(dims.begin(), dims.end(), static_cast<dim_t>(1),
                               std::multiplies<dim_t>());
    }

    template <rank_t r> basic_shape<r, dim_t> as_ranked() const
    {
        // TODO: use contracts of c++20
        if (r != rank()) { throw std::invalid_argument("invalid rank"); }
        std::array<dim_t, r> d;
        std::copy(dims.begin(), dims.end(), d.begin());
        return basic_shape<r, dim_t>(d);
    }

    const std::vector<dim_t> dims;
};
}  // namespace internal
}  // namespace ttl
