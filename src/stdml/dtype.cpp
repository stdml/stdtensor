#include <stdml/tensor.hpp>

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
    default:
        throw std::invalid_argument(__func__);
    }
#undef CASE
}
}  // namespace stdml
