#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <ttl/bits/std_except.hpp>
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

template <typename Dim = uint32_t>
class basic_flat_shape
{
    using dim_t = Dim;

    std::vector<dim_t> dims_;

  public:
    using dimension_type = Dim;

    template <typename... D>
    explicit basic_flat_shape(D... d) : dims_({static_cast<dim_t>(d)...})
    {
        static_assert(sizeof...(D) <= 0xff, "too many dimensions");
    }

    explicit basic_flat_shape(std::vector<dim_t> dims) : dims_(std::move(dims))
    {
    }

    template <rank_t r, typename D>
    explicit basic_flat_shape(const basic_shape<r, D> &shape)
        : dims_(std::move(arr2vec<Dim, D, r>(shape.dims())))
    {
    }

    rank_t rank() const { return static_cast<rank_t>(dims_.size()); }

    dim_t size() const { return product<dim_t>(dims_.begin(), dims_.end()); }

    basic_flat_shape subshape() const
    {
        if (dims_.size() < 1) {
            throw std::logic_error("scalar shape has no sub shape");
        }
        std::vector<dim_t> sub_dims(dims_.begin() + 1, dims_.end());
        return basic_flat_shape(std::move(sub_dims));
    }

    std::pair<dim_t, basic_flat_shape> uncons() const
    {
        if (dims_.size() < 1) {
            throw std::logic_error("scalar shape has no sub shape");
        }
        std::vector<dim_t> sub_dims(dims_.begin() + 1, dims_.end());
        return std::make_pair(dims_[0], basic_flat_shape(std::move(sub_dims)));
    }

    // experimental API!
    basic_flat_shape batch_shape(Dim n) const
    {
        std::vector<dim_t> batch_dims(dims_.size() + 1);
        batch_dims[0] = n;
        std::copy(dims_.begin(), dims_.end(), batch_dims.begin() + 1);
        return basic_flat_shape(std::move(batch_dims));
    }

    template <rank_t r>
    basic_shape<r, dim_t> ranked() const
    {
        if (r != rank()) { throw invalid_rank_reification(dims_, r); }
        std::array<dim_t, r> dims;
        std::copy(dims_.begin(), dims_.end(), dims.begin());
        return basic_shape<r, dim_t>(std::move(dims));
    }

    const std::vector<dim_t> &dims() const { return dims_; }

    bool operator==(const basic_flat_shape &s) const
    {
        if (dims_.size() != s.dims_.size()) { return false; }
        return std::equal(dims_.begin(), dims_.end(), s.dims().begin());
    }

    bool operator!=(const basic_flat_shape &s) const { return !operator==(s); }
};
}  // namespace internal
}  // namespace ttl
