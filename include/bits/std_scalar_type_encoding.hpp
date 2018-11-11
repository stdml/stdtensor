#pragma once
#include <cstdint>
#include <tuple>

namespace ttl
{
namespace internal
{
namespace idx_format
{
// http://yann.lecun.com/exdb/mnist/
template <typename R> struct data_type_t;

using V = std::uint8_t;

template <> struct data_type_t<std::uint8_t> {
    static constexpr V value = 0x08;
};

template <> struct data_type_t<std::int8_t> {
    static constexpr V value = 0x09;
};

template <> struct data_type_t<std::int16_t> {
    static constexpr V value = 0x0B;
};

template <> struct data_type_t<std::int32_t> {
    static constexpr V value = 0x0C;
};

template <> struct data_type_t<float> {
    static constexpr V value = 0x0D;
};

template <> struct data_type_t<double> {
    static constexpr V value = 0x0E;
};

struct encoding {
    using types = std::tuple<std::uint8_t, std::int8_t, std::int16_t,
                             std::int32_t, float, double>;

    using data_type = V;
    template <typename R> static constexpr data_type value()
    {
        return data_type_t<R>::value;
    }
};
}  // namespace idx_format
}  // namespace internal
}  // namespace ttl
