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
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_view;

template <typename R, typename shape_t> class basic_cuda_tensor<R, 0, shape_t>
{
    using allocator = cuda_mem_allocator<R>;

    using D = std::unique_ptr<R[], cuda_mem_deleter>;
    // using parent = base_tensor<R, shape_t, D>;

    const shape_t shape_;
    D data_;

  public:
    explicit basic_cuda_tensor(const shape_t &_) : basic_cuda_tensor() {}

    explicit basic_cuda_tensor() : data_(allocator()(shape_.size())) {}

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
};

template <typename R, typename shape_t>
class basic_cuda_tensor_ref<R, 0, shape_t>
    : public base_scalar<R, shape_t, ref_ptr<R>>
{
    using parent = base_scalar<R, shape_t, ref_ptr<R>>;
    using parent::parent;
};

template <typename R, typename shape_t>
class basic_cuda_tensor_view<R, 0, shape_t>
    : public base_scalar<R, shape_t, view_ptr<R>>
{
    using parent = base_scalar<R, shape_t, view_ptr<R>>;
    using parent::parent;
};

template <typename R, typename S, typename D>
class base_cuda_tensor : public base_tensor<R, S, D>
{
  protected:
    using parent = base_tensor<R, S, D>;
    using parent::parent;

    using parent::data_size;

  public:
    using parent::data;

    void from_host(const void *buffer) const
    {
        cudaMemcpy(data(), buffer, data_size(), cudaMemcpyHostToDevice);
    }

    void to_host(void *buffer) const
    {
        cudaMemcpy(buffer, data(), data_size(), cudaMemcpyDeviceToHost);
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor : public base_cuda_tensor<R, shape_t, ref_ptr<R>>
{
    using allocator = cuda_mem_allocator<R>;
    using Own = std::unique_ptr<R[], cuda_mem_deleter>;

    using parent = base_cuda_tensor<R, shape_t, ref_ptr<R>>;

    using self_t = basic_cuda_tensor<R, r, shape_t>;
    using ref_t = basic_cuda_tensor_ref<R, r, shape_t>;
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_cuda_tensor_ref<R, r - 1, subshape_shape_t>;

    Own data_owner_;

    explicit basic_cuda_tensor(R *data, const shape_t &shape)
        : parent(data, shape)
    {
    }

  public:
    template <typename... D>
    explicit basic_cuda_tensor(D... d) : basic_cuda_tensor(shape_t(d...))
    {
    }

    explicit basic_cuda_tensor(const shape_t &shape)
        : basic_cuda_tensor(allocator()(shape.size()), shape)
    {
    }

    subspace_t operator[](int i) const
    {
        return this->template _bracket<subspace_t>(i);
    }

    ref_t slice(int i, int j) const
    {
        return this->template _slice<ref_t>(i, j);
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_ref : public base_cuda_tensor<R, shape_t, ref_ptr<R>>
{
    using parent = base_cuda_tensor<R, shape_t, ref_ptr<R>>;
    using self_t = basic_cuda_tensor_ref<R, r, shape_t>;
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_cuda_tensor_ref<R, r - 1, subshape_shape_t>;

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
class basic_cuda_tensor_view : public base_cuda_tensor<R, shape_t, view_ptr<R>>
{
    using parent = base_cuda_tensor<R, shape_t, view_ptr<R>>;
    using self_t = basic_cuda_tensor_view<R, r, shape_t>;
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_cuda_tensor_view<R, r - 1, subshape_shape_t>;
    // using iter_t =
    //     basic_tensor_iterator<R, r - 1, subshape_shape_t, subspace_t>;

    using parent::from_host;  // disable

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

    // iter_t begin() const { return this->template _iter<iter_t>(this->data());
    // }

    // iter_t end() const
    // {
    //     return this->template _iter<iter_t>(this->data_end());
    // }

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
