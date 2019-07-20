#include "testing.hpp"

#include <ttl/bits/std_range.hpp>

using ttl::internal::range;

void test_range_n(int n)
{
    int s = 0;
    for (auto i : range(n)) { s += i; }
    ASSERT_EQ(s, n * (n - 1) / 2);
}

TEST(range_test, test_1)
{
    test_range_n(0);
    test_range_n(1);
    test_range_n(10);
    test_range_n(100);
    test_range_n(255);
    test_range_n(256);
}
