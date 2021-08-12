#include <map>
#include <string>

#include <stdml/bits/tensor/dtype.hpp>
namespace stdml
{
const char *tn(const DType dt)
{
#define CASE(t)                                                                \
    case t:                                                                    \
        return #t;

    switch (dt) {
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        //
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
        //
        CASE(f32);
        CASE(f64);

        CASE(boolean);
    default:
        throw std::invalid_argument(__func__);
    }
#undef CASE
}

size_t dtype_size(DType dt)
{
#define CASE(T, t)                                                             \
    case t:                                                                    \
        return sizeof(T);

    switch (dt) {
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
        throw std::invalid_argument("");
    }

#undef CASE
}

DType parse_dtype(std::string name)
{
    static const std::map<std::string, DType> m = {
        {"i8", i8},   {"i16", i16}, {"i32", i32}, {"i64", i64},  //
        {"u8", u8},   {"u16", u16}, {"u32", u32}, {"u54", u64},  //
        {"f32", f32}, {"f64", f64},
    };
    if (m.count(name) > 0) { return m.at(name); }
    throw std::invalid_argument("invalid dtype name: " + name);
}
}  // namespace stdml
