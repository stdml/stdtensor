#include <cstdlib>
#include <stdexcept>
#include <stdml/bits/tensor/allocator.hpp>

extern "C" {
void *cuda_alloc(size_t);
void cuda_free(void *);
}

namespace stdml
{
void *generic_allocator::alloc(Device device, size_t size)
{
    switch (device) {
    case cpu:
        return ::malloc(size);
    case cuda:
        return cuda_alloc(size);
    default:
        throw std::runtime_error("invalid device!");
    }
}

void generic_allocator::free(Device device, void *addr)
{
    // printf("generic_allocator::free(%s, %p)\n", device_name(device), addr);
    switch (device) {
    case cpu:
        ::free(addr);
        return;
    case cuda:
        cuda_free(addr);
        return;
    default:
        throw std::runtime_error("invalid device!");
    }
}
}  // namespace stdml
