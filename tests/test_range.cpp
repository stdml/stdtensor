#include "testing.hpp"

#include <ttl/algorithm>
#include <ttl/range>
#include <ttl/tensor>

using ttl::range;

int tri(int n) { return n * (n + 1) / 2; }

void test_range_n(int n)
{
    int s = 0;
    for (auto i : range(n)) { s += i; }
    ASSERT_EQ(s, tri(n - 1));
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

TEST(range_test, test_2)
{
    ttl::tensor<int, 3> x(4, 5, 6);
    int idx = 0;
    for (auto i : range<0>(x)) {
        for (auto j : range<1>(x)) {
            for (auto k : range<2>(x)) { x.at(i, j, k) = ++idx; }
        }
    }
    ASSERT_EQ(ttl::sum(view(x)), tri(4 * 5 * 6));
}
