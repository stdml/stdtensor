#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>

#include <stdml/bits/tensor/device.hpp>

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

Device parse_device(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [](char c) { return std::tolower(c); });
    static const std::map<std::string, Device> m = {
        {"cpu", cpu},
        {"gpu", cuda},
        {"cuda", cuda},
    };
    if (m.count(name) > 0) { return m.at(name); }
    throw std::invalid_argument("invalid device name: " + name);
}
}  // namespace stdml
