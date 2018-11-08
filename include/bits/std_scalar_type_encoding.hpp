#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ttl
{
namespace internal
{

using data_type_t = std::uint8_t;

namespace idx_format
{
namespace encoding
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
}  // namespace encoding

struct type_encoder {
    using data_type = data_type_t;

    template <typename R> static constexpr data_type value()
    {
        return encoding::data_type<R>::value;
    }

    static std::size_t size(const data_type type)
    {
        switch (type) {
        case value<std::uint8_t>():
            return sizeof(std::uint8_t);
        case value<std::int8_t>():
            return sizeof(std::int8_t);
        case value<std::int16_t>():
            return sizeof(std::int16_t);
        case value<std::int32_t>():
            return sizeof(std::int32_t);
        case value<float>():
            return sizeof(float);
        case value<double>():
            return sizeof(double);
        default:
            throw std::invalid_argument("invalid scalar code");
        }
    }
};

}  // namespace idx_format

using default_scalar_type_encoder = idx_format::type_encoder;

};  // namespace internal

}  // namespace ttl
