#pragma once
#include <stdexcept>

#ifndef USE_FAKE_CUDA_RUNTIME
#include <cuda_runtime.h>
#endif

namespace ttl
{
namespace internal
{
template <typename R> struct cuda_mem_allocator {
    R *operator()(int count)
    {
        R *deviceMem;
        // cudaMalloc<R>(&deviceMem, count);
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
        const cudaError_t err = cudaMalloc(&deviceMem, count * sizeof(R));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

}  // namespace internal
}  // namespace ttl
