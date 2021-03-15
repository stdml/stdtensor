#include <stdexcept>
#include <stdml/bits/device.hpp>

namespace stdml
{
const char *device_name(const Device d)
{
    switch (d) {
    case cpu:
        return "CPU";
    case cuda:
        return "CUDA";
    default:
        throw std::invalid_argument(__func__);
    }
}
}  // namespace stdml
