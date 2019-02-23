#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#ifndef USE_FAKE_CUDA_RUNTIME
#include <cuda_runtime.h>
#endif

#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{
template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        cudaMalloc<T>(&deviceMem, count);
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor;
template <typename R, rank_t r, typename shape_t> class basic_cuda_tensor_ref;

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor
{
    using self_t = basic_cuda_tensor<R, r, shape_t>;
    using ref_t = basic_cuda_tensor_ref<R, r, shape_t>;

    const shape_t shape_;
    const std::unique_ptr<R, cuda_mem_deleter> data_;

  public:
    template <typename... D>
    explicit basic_cuda_tensor(D... d)
        : shape_(d...), data_(cuda_mem_allocator<R>()(shape_.size()))
    {
    }

    R *data() { return data_.get(); }

    shape_t shape() const { return shape_; }

    void fromHost(void *buffer)
    {
        cudaMemcpy(data_.get(), buffer, shape_.size() * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void toHost(void *buffer)
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
class basic_cuda_tensor_ref
{
    using self_t = basic_cuda_tensor_ref<R, r, shape_t>;

    const shape_t shape_;
    R *const data_;

  public:
    template <typename... D>
    constexpr explicit basic_cuda_tensor_ref(R *data, D... d)
        : basic_cuda_tensor_ref(data, shape_t(d...))
    {
    }

    constexpr explicit basic_cuda_tensor_ref(R *data, const shape_t &shape)
        : shape_(shape), data_(data)
    {
    }

    R *data() const { return data_; }

    shape_t shape() const { return shape_; }

    void fromHost(void *buffer)
    {
        cudaMemcpy(data_, buffer, shape_.size() * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void toHost(void *buffer)
    {
        cudaMemcpy(buffer, data_, shape_.size() * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

    self_t slice(typename shape_t::dimension_type i,
                 typename shape_t::dimension_type j) const
    {
        const auto sub_shape = shape_.subshape();
        return self_t(data_ + i * sub_shape.size(), batch(j - i, sub_shape));
    }
};

}  // namespace internal
}  // namespace ttl
