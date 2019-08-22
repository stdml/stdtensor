#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_base_tensor.hpp>
#include <ttl/bits/std_cuda_allocator.hpp>
#include <ttl/bits/std_shape.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
template <typename T> class base_cuda_tensor
{
  public:
    void from_host(const void *buffer) const
    {
        cuda_copier::copy<cuda_copier::h2d>(
            static_cast<const T *>(this)->data(), buffer,
            static_cast<const T *>(this)->data_size());
    }

    void to_host(void *buffer) const
    {
        cuda_copier::copy<cuda_copier::d2h>(
            buffer, static_cast<const T *>(this)->data(),
            static_cast<const T *>(this)->data_size());
    }
};

template <typename R, typename shape_t>
class basic_cuda_tensor<R, 0, shape_t>
    : public base_scalar<R, shape_t, owner, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor<R, 0, shape_t>>
{
    using parent = base_scalar<R, shape_t, owner, basic_cuda_tensor_ref>;

    using allocator = cuda_mem_allocator<R>;
    using data_owner_t = std::unique_ptr<R, cuda_mem_deleter>;

    data_owner_t data_owner_;

    basic_cuda_tensor(R *data) : parent(data), data_owner_(data) {}

  public:
    explicit basic_cuda_tensor(const shape_t &_) : basic_cuda_tensor() {}

    explicit basic_cuda_tensor() : basic_cuda_tensor(allocator()(1)) {}
};

template <typename R, typename shape_t>
class basic_cuda_tensor_ref<R, 0, shape_t>
    : public base_scalar<R, shape_t, readwrite, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor_ref<R, 0, shape_t>>
{
    using parent = base_scalar<R, shape_t, readwrite, basic_cuda_tensor_ref>;
    using parent::parent;

  public:
    basic_cuda_tensor_ref(const basic_cuda_tensor<R, 0, shape_t> &t)
        : parent(t.data())
    {
    }
};

template <typename R, typename shape_t>
class basic_cuda_tensor_view<R, 0, shape_t>
    : public base_scalar<R, shape_t, readonly, basic_cuda_tensor_view>,
      public base_cuda_tensor<basic_cuda_tensor_view<R, 0, shape_t>>
{
    using parent = base_scalar<R, shape_t, readonly, basic_cuda_tensor_view>;
    using parent::parent;

    using mixin = base_cuda_tensor<basic_cuda_tensor_view<R, 0, shape_t>>;
    using mixin::from_host;  // disable

  public:
    basic_cuda_tensor_view(const basic_cuda_tensor<R, 0, shape_t> &t)
        : parent(t.data())
    {
    }

    basic_cuda_tensor_view(const basic_cuda_tensor_ref<R, 0, shape_t> &t)
        : parent(t.data())
    {
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor
    : public base_tensor<R, shape_t, owner, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor<R, r, shape_t>>
{
    using parent = base_tensor<R, shape_t, owner, basic_cuda_tensor_ref>;
    using allocator = cuda_mem_allocator<R>;
    using data_owner_t = std::unique_ptr<R[], cuda_mem_deleter>;

    data_owner_t data_owner_;

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
    : public base_tensor<R, shape_t, readwrite, basic_cuda_tensor_ref>,
      public base_cuda_tensor<basic_cuda_tensor_ref<R, r, shape_t>>
{
    using parent = base_tensor<R, shape_t, readwrite, basic_cuda_tensor_ref>;

  public:
    template <typename... D>
    constexpr explicit basic_cuda_tensor_ref(R *data, D... d)
        : basic_cuda_tensor_ref(data, shape_t(d...))
    {
    }

    basic_cuda_tensor_ref(const basic_cuda_tensor<R, r, shape_t> &t)
        : basic_cuda_tensor_ref(t.data(), t.shape())
    {
    }

    constexpr explicit basic_cuda_tensor_ref(R *data, const shape_t &shape)
        : parent(data, shape)
    {
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor_view
    : public base_tensor<R, shape_t, readonly, basic_cuda_tensor_view>,
      public base_cuda_tensor<basic_cuda_tensor_view<R, r, shape_t>>
{
    using parent = base_tensor<R, shape_t, readonly, basic_cuda_tensor_view>;
    using mixin = base_cuda_tensor<basic_cuda_tensor_view<R, r, shape_t>>;
    using mixin::from_host;  // disable

  public:
    template <typename... D>
    constexpr explicit basic_cuda_tensor_view(const R *data, D... d)
        : basic_cuda_tensor_view(data, shape_t(d...))
    {
    }

    basic_cuda_tensor_view(const basic_cuda_tensor<R, r, shape_t> &t)
        : basic_cuda_tensor_view(t.data(), t.shape())
    {
    }

    basic_cuda_tensor_view(const basic_cuda_tensor_ref<R, r, shape_t> &t)
        : basic_cuda_tensor_view(t.data(), t.shape())
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
