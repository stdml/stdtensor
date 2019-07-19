#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_base_tensor.hpp>
#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{

template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_ref;

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor
// : public base_tensor<R, shape_t, std::unique_ptr<R[], cuda_mem_deleter>>
{
    using allocator = cuda_mem_allocator<R>;

    using D = std::unique_ptr<R[], cuda_mem_deleter>;
    using parent = base_tensor<R, shape_t, D>;

    using self_t = basic_cuda_tensor<R, r, shape_t>;
    using ref_t = basic_cuda_tensor_ref<R, r, shape_t>;

    const shape_t shape_;
    D data_;

    // using parent::data_;
    // using parent::shape_;

    // explicit basic_cuda_tensor(const shape_t &shape, allocator &alloc)
    //     : parent(D(alloc(shape.size())), shape)
    // {
    // }

  public:
    template <typename... D>
    explicit basic_cuda_tensor(D... d) : basic_cuda_tensor(shape_t(d...))
    {
    }

    explicit basic_cuda_tensor(const shape_t &shape)
        : shape_(shape), data_(cuda_mem_allocator<R>()(shape.size()))
    {
    }

    R *data() const { return data_.get(); }

    R *data_end() const { return data_.get() + shape().size(); }

    shape_t shape() const { return shape_; }

    void from_host(const void *buffer) const
    {
        cudaMemcpy(data_.get(), buffer, shape_.size() * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void to_host(void *buffer) const
    {
        cudaMemcpy(buffer, data_.get(), shape_.size() * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

    ref_t slice(typename shape_t::dimension_type i,
                typename shape_t::dimension_type j) const
    {
        const auto sub_shape = shape_.subshape();
        return ref_t(data_.get() + i * sub_shape.size(),
                     batch(j - i, sub_shape));
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_ref : public base_tensor<R, shape_t, ref_ptr<R>>
{
    using parent = base_tensor<R, shape_t, ref_ptr<R>>;
    using self_t = basic_cuda_tensor_ref<R, r, shape_t>;

    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_cuda_tensor_ref<R, r - 1, subshape_shape_t>;

    using parent::data_;
    using parent::shape_;

  public:
    template <typename... D>
    constexpr explicit basic_cuda_tensor_ref(R *data, D... d)
        : basic_cuda_tensor_ref(data, shape_t(d...))
    {
    }

    constexpr explicit basic_cuda_tensor_ref(R *data, const shape_t &shape)
        : parent(data, shape)
    {
    }

    void from_host(const void *buffer) const
    {
        cudaMemcpy(data_, buffer, shape_.size() * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void to_host(void *buffer) const
    {
        cudaMemcpy(buffer, data_, shape_.size() * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

    subspace_t operator[](int i) const
    {
        return this->template _bracket<subspace_t>(i);
    }

    self_t slice(int i, int j) const
    {
        return this->template _slice<self_t>(i, j);
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_view : public base_tensor<R, shape_t, view_ptr<R>>
{
    using parent = base_tensor<R, shape_t, view_ptr<R>>;
    using self_t = basic_cuda_tensor_view<R, r, shape_t>;

    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_cuda_tensor_view<R, r - 1, subshape_shape_t>;

    using parent::data_;
    using parent::shape_;

  public:
    template <typename... D>
    constexpr explicit basic_cuda_tensor_view(const R *data, D... d)
        : basic_cuda_tensor_view(data, shape_t(d...))
    {
    }

    constexpr explicit basic_cuda_tensor_view(const R *data,
                                              const shape_t &shape)
        : parent(data, shape)
    {
    }

    void to_host(void *buffer) const
    {
        cudaMemcpy(buffer, data_, shape_.size() * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

    subspace_t operator[](int i) const
    {
        return this->template _bracket<subspace_t>(i);
    }

    self_t slice(int i, int j) const
    {
        return this->template _slice<self_t>(i, j);
    }
};

}  // namespace internal
}  // namespace ttl
