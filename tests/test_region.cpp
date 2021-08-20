#include "testing.hpp"

#include <ttl/experimental/region>

TEST(range_test, test_1)
{
    {
        ttl::experimental::region_t r(1, 10);
        int x = 0;
        for (auto i : ttl::experimental::range_t(r)) { x += i; }
        ASSERT_EQ(x, 55);
    }
    {
        ttl::experimental::region_t r(0, 10);
        int x = 0;
        for (auto i : ttl::experimental::range_t(r)) { x += i; }
        ASSERT_EQ(x, 45);
    }
}
