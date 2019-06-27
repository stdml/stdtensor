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

TEST(shape_debug_test, test1)
{
    {
        const auto s = ttl::make_shape();
        test_shape_to_string("()", s);
    }
    {
        const auto s = ttl::make_shape(2);
        test_shape_to_string("(2)", s);
    }
    {
        const auto s = ttl::make_shape(2, 3);
        test_shape_to_string("(2, 3)", s);
    }
    {
        const auto s = ttl::make_shape(2, 3, 4);
        test_shape_to_string("(2, 3, 4)", s);
    }
}
