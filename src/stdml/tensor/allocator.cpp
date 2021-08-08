#include <cstdlib>
#include <stdexcept>
#include <stdml/bits/dll.hpp>
#include <stdml/bits/tensor/allocator.hpp>
#include <stdml/bits/tensor/cudart.hpp>

namespace stdml
{
bool has_cuda() { return try_dl_open("/usr/local/cuda/lib64/libcudart.so"); }

void *generic_allocator::alloc(Device device, size_t size)
{
    switch (device) {
    case cpu:
        return ::malloc(size);
    case cuda:
        return libcudart::get().cuda_alloc(size);
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
        libcudart::get().cuda_free(addr);
        return;
    default:
        throw std::runtime_error("invalid device!");
    }
}

template <Device Dst, Device Src>
void generic_copier<Dst, Src>::operator()(void *dst, const void *src,
                                          size_t size) const
{
}
}  // namespace stdml
