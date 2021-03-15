#pragma once
#include <cstdint>

namespace stdml
{
enum Device : uint8_t {
    cpu,
    cuda,
};

extern const char *device_name(const Device d);

inline constexpr Device gpu = cuda;
}  // namespace stdml
