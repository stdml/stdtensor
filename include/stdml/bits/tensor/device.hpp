#pragma once
#include <cstdint>
#include <string>

#include <ttl/device>

namespace stdml
{
enum Device : uint8_t {
    cpu,
    cuda,
};

extern const char *device_name(const Device d);
Device parse_device(std::string);

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

template <Device>
struct device_t;

template <>
struct device_t<cpu> {
    using type = ttl::host_memory;
};

template <>
struct device_t<cuda> {
    using type = ttl::cuda_memory;
};
}  // namespace stdml
