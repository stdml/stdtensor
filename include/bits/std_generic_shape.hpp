#pragma once
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include <bits/std_shape.hpp>

namespace ttl
{
namespace internal
{

template <typename dim_t = uint32_t> class basic_generic_shape
{
  public:
    using dim_type = dim_t;

    template <typename... D>
    explicit basic_generic_shape(D... d) : dims({static_cast<dim_t>(d)...})
    {
        static_assert(sizeof...(D) <= 0xff, "too many dimensions");
    }

    explicit basic_generic_shape(const std::vector<dim_t> &dims) : dims(dims) {}

    rank_t rank() const { return dims.size(); }

    dim_t size() const
    {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               std::multiplies<dim_t>());
    }

    template <rank_t r> basic_shape<r, dim_t> as_ranked() const
    {
        // TODO: use contracts of c++20
        assert(r == rank());
        std::array<dim_t, r> d;
        std::copy(dims.begin(), dims.end(), d.begin());
        return basic_shape<r, dim_t>(d);
    }

    const std::vector<dim_t> dims;
};

}  // namespace internal
}  // namespace ttl
