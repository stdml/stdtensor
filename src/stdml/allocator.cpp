#include <cstdlib>
#include <stdexcept>
#include <stdml/bits/tensor/allocator.hpp>

namespace stdml
{
void *generic_allocator::alloc(Device device, size_t size)
{
    if (device == cpu) {
        return ::malloc(size);
    } else {
        throw std::runtime_error("TODO: !");
    }
}

void generic_allocator::free(Device device, void *addr)
{
    if (device == cpu) {
        ::free(addr);
    } else {
        throw std::runtime_error("TODO: !");
    }
}
}  // namespace stdml
