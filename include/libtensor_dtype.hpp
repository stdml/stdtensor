#pragma once
#include <cstdint>
#include <stdexcept>

namespace stdml
{
enum DType {
    i8,
    i16,
    i32,
    i64,

    //
    u8,
    u16,
    u32,
    u64,

    //
    bf16,
    f16,
    f32,
    f64,

    //
    boolean,
};

template <typename E>
DType from(typename E::value_type v)
{
#define CASE(T, t)                                                             \
    case E::template value<T>():                                               \
        return t;

    switch (v) {
        CASE(int8_t, i8);
        CASE(int16_t, i16);
        CASE(int32_t, i32);
        CASE(int64_t, i64);

        CASE(uint8_t, u8);
        CASE(uint16_t, u16);
        CASE(uint32_t, u32);
        CASE(uint64_t, u64);

        CASE(float, f32);
        CASE(double, f64);
    default:
        throw std::invalid_argument(__func__);
    }
#undef CASE
}

template <typename E>
auto to(DType v)
{
#define CASE(T, t)                                                             \
    case t:                                                                    \
        return E::template value<T>();

    switch (v) {
        CASE(int8_t, i8);
        CASE(int16_t, i16);
        CASE(int32_t, i32);
        CASE(int64_t, i64);

        CASE(uint8_t, u8);
        CASE(uint16_t, u16);
        CASE(uint32_t, u32);
        CASE(uint64_t, u64);

        CASE(float, f32);
        CASE(double, f64);
    default:
        throw std::invalid_argument(__func__);
    }
#undef CASE
}
}  // namespace stdml
