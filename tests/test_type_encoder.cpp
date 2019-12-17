#include "testing.hpp"
#include <experimental/type_encoder>
#include <ttl/bits/idx_encoding.hpp>

TEST(type_encoder_test, test1)
{
    using encoder = std::experimental::basic_type_encoder<
        ttl::internal::idx_format::encoding>;

    using V = encoder::value_type;

    ASSERT_EQ(encoder::value<uint8_t>(), static_cast<V>(8));
    ASSERT_EQ(encoder::value<int8_t>(), static_cast<V>(9));
    ASSERT_EQ(encoder::value<int16_t>(), static_cast<V>(11));
    ASSERT_EQ(encoder::value<int32_t>(), static_cast<V>(12));
    ASSERT_EQ(encoder::value<float>(), static_cast<V>(13));
    ASSERT_EQ(encoder::value<double>(), static_cast<V>(14));

    ASSERT_EQ(encoder::size(8), static_cast<size_t>(1));   // u8
    ASSERT_EQ(encoder::size(9), static_cast<size_t>(1));   // i8
    ASSERT_EQ(encoder::size(11), static_cast<size_t>(2));  // i16
    ASSERT_EQ(encoder::size(12), static_cast<size_t>(4));  // i32
    ASSERT_EQ(encoder::size(13), static_cast<size_t>(4));  // f32
    ASSERT_EQ(encoder::size(14), static_cast<size_t>(8));  // f64
}
