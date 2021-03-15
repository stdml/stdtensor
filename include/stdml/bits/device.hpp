#pragma once
#include <cstdint>
#include <ttl/device>

namespace stdml
{
enum Device : uint8_t {
    cpu,
    cuda,
};

extern const char *device_name(const Device d);

inline constexpr Device gpu = cuda;

template <typename>
struct device_type;

template <>
struct device_type<ttl::host_memory> {
    static constexpr Device value = cpu;
};

template <>
struct device_type<ttl::cuda_memory> {
    static constexpr Device value = cuda;
};
}  // namespace stdml
