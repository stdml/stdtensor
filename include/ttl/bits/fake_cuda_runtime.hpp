#pragma once
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

using cudaError_t = int;
using cudaMemcpyKind = int;

constexpr const cudaError_t cudaSuccess = 0;

constexpr const cudaMemcpyKind cudaMemcpyHostToDevice = 1;
constexpr const cudaMemcpyKind cudaMemcpyDeviceToHost = 2;
constexpr const cudaMemcpyKind cudaMemcpyDeviceToDevice = 3;

class fake_cuda_device
{
    std::map<const void *, size_t> _allocs;

    void check_leak() const
    {
        if (!_allocs.empty()) {
            throw std::runtime_error("device memory leak detected.");
        }
    }

    void check_alloc(const void *data, size_t size) const
    {
        const auto pos = _allocs.find(data);
        if (pos == _allocs.end()) { throw std::runtime_error("not allocated"); }
        if (pos->second != size) {
            throw std::runtime_error("alloc size not match");
        }
    }

  public:
    fake_cuda_device() { std::printf("using fake_cuda_device!\n"); }

    ~fake_cuda_device() { check_leak(); }

    void *alloc(size_t size)
    {
        void *ptr = std::malloc(size);
        _allocs[ptr] = size;
        return ptr;
    }

    void free(void *data)
    {
        if (_allocs.count(data) == 0) {
            throw std::runtime_error("invalid free");
        }
        std::free(data);
        _allocs.erase(data);
    }

    void memcpy(void *dst, const void *src, size_t size,
                cudaMemcpyKind dir) const
    {
        switch (dir) {
        case cudaMemcpyHostToDevice:
            check_alloc(dst, size);
            break;
        case cudaMemcpyDeviceToHost:
            check_alloc(src, size);
            break;
        default:
            throw std::runtime_error("invalid memcpy direction");
        }
        std::memcpy(dst, src, size);
    }
};

fake_cuda_device fake_cuda;

cudaError_t cudaMalloc(void **ptr, size_t count)
{
    *ptr = fake_cuda.alloc(count);
    return cudaSuccess;
}

cudaError_t cudaFree(void *ptr)
{
    fake_cuda.free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
                       cudaMemcpyKind dir)
{
    fake_cuda.memcpy(dst, src, size, dir);
    return cudaSuccess;
}

std::string cudaGetErrorString(const cudaError_t err)
{
    return "fake_cudaError_t(" + std::to_string(static_cast<int>(err)) + ")";
}
