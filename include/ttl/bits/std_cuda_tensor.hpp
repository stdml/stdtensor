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

template <typename T> class base_cuda_tensor
{
  public:
    void from_host(const void *buffer) const
    {
        void *data = static_cast<const T *>(this)->data();
        const auto size = static_cast<const T *>(this)->data_size();
        cudaMemcpy(data, buffer, size, cudaMemcpyHostToDevice);
    }

    void to_host(void *buffer) const
    {
        const void *data = static_cast<const T *>(this)->data();
        const auto size = static_cast<const T *>(this)->data_size();
        cudaMemcpy(buffer, data, size, cudaMemcpyDeviceToHost);
    }
};

template <typename R, typename shape_t>
class basic_cuda_tensor<R, 0, shape_t>
    : public base_scalar<R, shape_t, ref_ptr<R>>,
      public base_cuda_tensor<basic_cuda_tensor<R, 0, shape_t>>
{
    using allocator = cuda_mem_allocator<R>;
    using owner = std::unique_ptr<R, cuda_mem_deleter>;

    using parent = base_scalar<R, shape_t, ref_ptr<R>>;

    owner data_owner_;

    basic_cuda_tensor(R *data) : parent(data), data_owner_(data) {}

  public:
    explicit basic_cuda_tensor(const shape_t &_) : basic_cuda_tensor() {}

    explicit basic_cuda_tensor() : basic_cuda_tensor(allocator()(1)) {}
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

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor
    : public base_tensor<R, shape_t, ref_ptr<R>, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor<R, r, shape_t>>
{
    using allocator = cuda_mem_allocator<R>;
    using owner = std::unique_ptr<R[], cuda_mem_deleter>;

    using parent = base_tensor<R, shape_t, ref_ptr<R>, basic_cuda_tensor_ref>;

    owner data_owner_;

    explicit basic_cuda_tensor(R *data, const shape_t &shape)
        : parent(data, shape), data_owner_(data)
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
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_ref
    : public base_tensor<R, shape_t, ref_ptr<R>, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor_ref<R, r, shape_t>>
{
    using parent = base_tensor<R, shape_t, ref_ptr<R>, basic_cuda_tensor_ref>;

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
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_view
    : public base_tensor<R, shape_t, view_ptr<R>, basic_cuda_tensor_view>,
      public base_cuda_tensor<basic_cuda_tensor_view<R, r, shape_t>>
{
    using parent = base_tensor<R, shape_t, view_ptr<R>, basic_cuda_tensor_view>;
    using mixin = base_cuda_tensor<basic_cuda_tensor_view<R, r, shape_t>>;
    using mixin::from_host;  // disable

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
};

}  // namespace internal
}  // namespace ttl
