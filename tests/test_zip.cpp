#include "testing.hpp"

#include <vector>

#include <ttl/experimental/zip>
#include <ttl/range>

using ttl::range;
using ttl::experimental::zip;

#if defined(__clang__)
// /*
TEST(zip_test, test1)
{
    const int n = 10;
    int s = 0;
    // for (auto [i, j] : zip(range(n), range(n))) { s += i + j; }
    for (auto p : zip(range(n), range(n))) {
        auto i = std::get<0>(p);
        auto j = std::get<1>(p);
        s += i + j;
    }
    ASSERT_EQ(s, 90);
}

TEST(zip_test, test2)
{
    const int n = 10;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    int s = 0;
    // for (auto [i, j] : zip(range(n), v)) { s += i + j; }
    for (auto p : zip(range(n), v)) {
        auto i = std::get<0>(p);
        auto j = std::get<1>(p);
        s += i + j;
    }
    ASSERT_EQ(s, 90);
}
// */
#endif

TEST(zip_test, test3)
{
    const int n = 10;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    int s = 0;
    for (auto p : zip(v, v)) {
        auto i = std::get<0>(p);
        auto j = std::get<1>(p);
        s += i + j;
    }
    ASSERT_EQ(s, 90);
}
