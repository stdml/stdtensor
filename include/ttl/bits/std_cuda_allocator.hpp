#pragma once
#include <stdexcept>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_cuda_runtime.hpp>

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

template <typename R>
using cuda_mem_allocator = basic_allocator<R, cuda_memory>;

struct cuda_mem_deleter {
    void operator()(void *ptr)
    {
        const cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) { throw std::runtime_error("cudaFree failed"); }
    }
};
}  // namespace internal
}  // namespace ttl
