#pragma once
#include <cstring>
#include <map>
#include <stdexcept>

class fake_device
{
    std::map<const void *, size_t> _allocs;

    void check_leak() const
    {
        if (not _allocs.empty()) {
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
    ~fake_device() { check_leak(); }

    void *alloc(size_t size)
    {
        void *ptr = malloc(size);
        _allocs[ptr] = size;
        return ptr;
    }

    void free(void *data)
    {
        if (_allocs.count(data) == 0) {
            throw std::runtime_error("invalid free");
        }
        _allocs.erase(data);
    }

    static constexpr int h2d = 1;
    static constexpr int d2h = 2;

    void memcpy(void *dst, const void *src, int size, int direction) const
    {
        switch (direction) {
        case h2d:
            check_alloc(dst, size);
            break;
        case d2h:
            check_alloc(src, size);
            break;
        default:
            throw std::runtime_error("invalid memcpy direction");
        }
        std::memcpy(dst, src, size);
    }
};

fake_device fake_cuda;

template <typename T> void cudaMalloc(T **ptr, int count)
{
    *ptr = (T *)fake_cuda.alloc(sizeof(T) * count);
}

void cudaFree(void *ptr) { fake_cuda.free(ptr); }

constexpr int cudaMemcpyHostToDevice = fake_device::h2d;
constexpr int cudaMemcpyDeviceToHost = fake_device::d2h;

void cudaMemcpy(void *dst, const void *src, int size, int direction)
{
    fake_cuda.memcpy(dst, src, size, direction);
}
