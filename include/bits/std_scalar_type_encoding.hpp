#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ttl
{
namespace internal
{

class scalar_info
{
  public:
    virtual size_t size() const = 0;
};

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

inline size_t get_size(data_type_t type)
{
    switch (type) {
    case data_type<std::uint8_t>::value:
        return sizeof(std::uint8_t);
    case data_type<std::int8_t>::value:
        return sizeof(std::int8_t);
    case data_type<std::int16_t>::value:
        return sizeof(std::int16_t);
    case data_type<std::int32_t>::value:
        return sizeof(std::int32_t);
    case data_type<float>::value:
        return sizeof(float);
    case data_type<double>::value:
        return sizeof(double);
    default:
        throw std::invalid_argument("invalid scalar code");
    }
}

class scalar_info_impl : public scalar_info
{
    // const data_type_t type_;
    const size_t size_;

  public:
    scalar_info_impl(data_type_t type) : size_(get_size(type)) {}

    virtual size_t size() const override { return size_; }
};
}  // namespace idx_format

template <typename R>
// TODO: make it extendable
using default_scalar_type_encoding = idx_format::data_type<R>;

using default_scalar_info_t = idx_format::scalar_info_impl;

template <typename R> constexpr data_type_info typeinfo()
{
    using tt = default_scalar_type_encoding<R>;
    return data_type_info{tt::value, sizeof(R)};
}
};  // namespace internal

}  // namespace ttl
