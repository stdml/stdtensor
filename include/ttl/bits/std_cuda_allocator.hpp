#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

#include <ttl/bits/std_cuda_runtime.hpp>
#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>

namespace ttl
{
namespace internal
{
class std_cuda_error_checker_t
{
    const std::string func_name_;

  public:
    std_cuda_error_checker_t(const char *func_name) : func_name_(func_name) {}

    void operator<<(const cudaError_t err) const
    {
        if (err != cudaSuccess) {
            throw std::runtime_error(func_name_ + " failed with: " +
                                     std::to_string(static_cast<int>(err)) +
                                     ": " + cudaGetErrorString(err));
        }
    }
};  // namespace ttl

struct cuda_copier {
    static constexpr auto h2d = cudaMemcpyHostToDevice;
    static constexpr auto d2h = cudaMemcpyDeviceToHost;
    static constexpr auto d2d = cudaMemcpyDeviceToDevice;

    template <cudaMemcpyKind dir>
    static void copy(void *dst, const void *src, size_t size)
    {
        static std_cuda_error_checker_t check("cudaMemcpy");
        check << cudaMemcpy(dst, src, size, dir);
    }
};

template <>
class basic_copier<host_memory, cuda_memory>
{
  public:
    void operator()(void *dst, const void *src, size_t size)
    {
        cuda_copier::copy<cuda_copier::d2h>(dst, src, size);
    }
};

template <>
class basic_copier<cuda_memory, host_memory>
{
  public:
    void operator()(void *dst, const void *src, size_t size)
    {
        cuda_copier::copy<cuda_copier::h2d>(dst, src, size);
    }
};

template <typename R>
class basic_allocator<R, cuda_memory>
{
  public:
    R *operator()(size_t count)
    {
        void *deviceMem;
        // cudaMalloc<R>(&deviceMem, count);
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
        static std_cuda_error_checker_t check("cudaMalloc");
        check << cudaMalloc(&deviceMem, count * sizeof(R));
        return reinterpret_cast<R *>(deviceMem);
    }
};

template <typename R>
class basic_deallocator<R, cuda_memory>
{
  public:
    void operator()(R *data)
    {
        static std_cuda_error_checker_t check("cudaFree");
        check << cudaFree(data);
    }
};
}  // namespace internal
}  // namespace ttl
