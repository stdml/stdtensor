#pragma once
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
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
