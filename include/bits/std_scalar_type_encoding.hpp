#pragma once
#include <cstdint>

namespace ttl
{
namespace internal
{

struct data_type_info {
    const uint8_t code;
    const uint8_t size;
};

using data_type_t = std::uint8_t;

namespace idx_format
{
// http://yann.lecun.com/exdb/mnist/
template <typename R> struct data_type;

template <> struct data_type<std::uint8_t> {
    static constexpr data_type_t value = 0x08;
};

template <> struct data_type<std::int8_t> {
    static constexpr data_type_t value = 0x09;
};

template <> struct data_type<std::int16_t> {
    static constexpr data_type_t value = 0x0B;
};

template <> struct data_type<std::int32_t> {
    static constexpr data_type_t value = 0x0C;
};

template <> struct data_type<float> {
    static constexpr data_type_t value = 0x0D;
};

template <> struct data_type<double> {
    static constexpr data_type_t value = 0x0E;
};
}  // namespace idx_format

template <typename R>
// TODO: make it extendable
using default_scalar_type_encoding = idx_format::data_type<R>;

template <typename R> constexpr data_type_info typeinfo()
{
    using tt = default_scalar_type_encoding<R>;
    return data_type_info{tt::value, sizeof(R)};
}
};  // namespace internal

}  // namespace ttl
