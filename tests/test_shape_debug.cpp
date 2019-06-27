#include "testing.hpp"

#include <ttl/debug>
#include <ttl/tensor>

template <ttl::rank_t r, typename D>
void test_shape_to_string(const std::string &str,
                          const ttl::internal::basic_shape<r, D> &s)
{
    ASSERT_EQ(str, ttl::to_string(s));
    const auto rs = ttl::internal::basic_raw_shape<D>(s);
    ASSERT_EQ(str, ttl::to_string(rs));
}

template <typename D> void test_shape_to_string_all()
{
    {
        const auto s = ttl::internal::basic_shape<0, D>();
        test_shape_to_string("()", s);
    }
    {
        const auto s = ttl::internal::basic_shape<1, D>(2);
        test_shape_to_string("(2)", s);
    }
    {
        const auto s = ttl::internal::basic_shape<2, D>(2, 3);
        test_shape_to_string("(2, 3)", s);
    }
    {
        const auto s = ttl::internal::basic_shape<3, D>(2, 3, 4);
        test_shape_to_string("(2, 3, 4)", s);
    }
}

TEST(shape_debug_test, test1)
{
    // FIXME: support i8/u8

    // test_shape_to_string_all<int8_t>();
    test_shape_to_string_all<int16_t>();
    test_shape_to_string_all<int32_t>();
    test_shape_to_string_all<int64_t>();

    // test_shape_to_string_all<uint8_t>();
    test_shape_to_string_all<uint16_t>();
    test_shape_to_string_all<uint32_t>();
    test_shape_to_string_all<uint64_t>();
}
