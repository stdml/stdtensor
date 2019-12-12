#pragma once
#include <stdexcept>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_cuda_runtime.hpp>
#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
struct cuda_copier {
    static constexpr auto h2d = cudaMemcpyHostToDevice;
    static constexpr auto d2h = cudaMemcpyDeviceToHost;
    static constexpr auto d2d = cudaMemcpyDeviceToDevice;

    template <cudaMemcpyKind dir>
    static void copy(void *dst, const void *src, size_t size)
    {
        const cudaError_t err = cudaMemcpy(dst, src, size, dir);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed");
        }
    }
};

template <> class basic_copier<host_memory, cuda_memory>
{
  public:
    void operator()(void *dst, const void *src, size_t size)
    {
        cuda_copier::copy<cuda_copier::d2h>(dst, src, size);
    }
};

template <> class basic_copier<cuda_memory, host_memory>
{
  public:
    void operator()(void *dst, const void *src, size_t size)
    {
        cuda_copier::copy<cuda_copier::h2d>(dst, src, size);
    }
};

template <typename R> class basic_allocator<R, cuda_memory>
{
  public:
    R *operator()(int count)
    {
        void *deviceMem;
        // cudaMalloc<R>(&deviceMem, count);
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
        const cudaError_t err = cudaMalloc(&deviceMem, count * sizeof(R));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        return reinterpret_cast<R *>(deviceMem);
    }
};

template <typename R> class basic_deallocator<R, cuda_memory>
{
  public:
    void operator()(R *data)
    {
        const cudaError_t err = cudaFree(data);
        if (err != cudaSuccess) { throw std::runtime_error("cudaFree failed"); }
    }
};
}  // namespace internal
}  // namespace ttl
