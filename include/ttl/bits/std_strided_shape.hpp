#pragma once
#include <memory>

#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{

template <rank_t r, typename Dim = typename basic_shape<r>::dimension_type>
class basic_strided_shape
{
    using dim_t = Dim;
    using S = std::array<dim_t, r>;

    const S dims_;
    const S strides_;

  public:
    using dimension_type = Dim;

    explicit basic_strided_shape(const basic_shape<r, dim_t> &shape,
                                 const S &strides)
        : basic_strided_shape(shape.dims, strides)
    {
    }

    template <typename... D>
    explicit basic_strided_shape(const basic_shape<r, dim_t> &shape, D... d)
        : basic_strided_shape(shape.dims, S({static_cast<dim_t>(d)...}))
    {
        static_assert(sizeof...(D) == r, "invalid number of strides");
    }

    explicit basic_strided_shape(const S &dims, const S &strides)
        : dims_(dims), strides_(strides)
    {
    }

    template <typename... I> dim_t offset(I... args) const
    {
        static_assert(sizeof...(I) == r, "invalid number of indexes");

        // TODO: expand the expression
        const S offs({static_cast<dim_t>(args)...});
        dim_t off = 0;
        for (rank_t i = 0; i < r; ++i) { off = off * dims_[i] + offs[i]; }
        return off;
    }
};

}  // namespace internal
}  // namespace ttl
