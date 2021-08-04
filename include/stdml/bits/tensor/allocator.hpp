#pragma once
#include <memory>
#include <stdml/bits/tensor/device.hpp>

namespace stdml
{
bool has_cuda();

class generic_allocator
{
  public:
    static void *alloc(Device device, size_t size);

    static void free(Device device, void *addr);
};

template <Device Dst, Device Src>
class generic_copier
{
  public:
    void operator()(void *dst, const void *src, size_t size) const;
};

class generic_pointer
{
    using GA = generic_allocator;

    Device device_;
    void *addr_;

  public:
    generic_pointer(Device device, size_t size)
        : device_(device), addr_(GA::alloc(device, size))
    {
    }

    generic_pointer(void *addr, Device device) : device_(device), addr_(addr) {}

    ~generic_pointer() { GA::free(device_, addr_); }

    void *addr() { return addr_; }
};

class Buffer
{
    std::unique_ptr<generic_pointer> p_;

  public:
    Buffer(void *addr, Device device) : p_(new generic_pointer(addr, device)) {}

    Buffer(size_t size, Device device) : p_(new generic_pointer(device, size))
    {
    }

    void *data() const { return p_.get(); }
};
}  // namespace stdml
