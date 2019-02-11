#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#ifndef FAKE_CUDA_RUNTIME
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

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_cuda_tensor
{
    const shape_t shape_;
    const size_t count;
    const std::unique_ptr<R, cuda_mem_deleter> data_;

  public:
    template <typename... D>
    explicit basic_cuda_tensor(D... d)
        : shape_(d...), count(shape_.size()),
          data_(cuda_mem_allocator<R>()(count))
    {
    }

    R *data() { return data_.get(); }

    void fromHost(void *buffer)
    {
        cudaMemcpy(data_.get(), buffer, count * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void toHost(void *buffer)
    {
        cudaMemcpy(buffer, data_.get(), count * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }
};

template <typename R, rank_t r> using cuda_tensor = basic_cuda_tensor<R, r>;
}  // namespace internal
}  // namespace ttl
