#include "testing.hpp"
#include <bits/std_scalar_type_encoding.hpp>
#include <experimental/type_encoder>

TEST(type_encoder_test, test1)
{
    using encoder = std::experimental::basic_type_encoder<
        ttl::internal::idx_format::encoding>;

    ASSERT_EQ(encoder::value<uint8_t>(), 8);
    ASSERT_EQ(encoder::value<int8_t>(), 9);
    ASSERT_EQ(encoder::value<int16_t>(), 11);
    ASSERT_EQ(encoder::value<int32_t>(), 12);
    ASSERT_EQ(encoder::value<float>(), 13);
    ASSERT_EQ(encoder::value<double>(), 14);

    ASSERT_EQ(encoder::size(8), 1);   // u8
    ASSERT_EQ(encoder::size(9), 1);   // i8
    ASSERT_EQ(encoder::size(11), 2);  // i16
    ASSERT_EQ(encoder::size(12), 4);  // i32
    ASSERT_EQ(encoder::size(13), 4);  // f32
    ASSERT_EQ(encoder::size(14), 8);  // f64
}
