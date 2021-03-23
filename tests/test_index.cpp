#include "testing.hpp"

#include <algorithm>

#include <ttl/bits/std_index.hpp>
#include <ttl/tensor>

TEST(index_test, test_1)
{
    ttl::tensor<int, 2> x(10, 10);
    std::iota(x.data(), x.data_end(), 0);
    static_assert(std::is_same<decltype(x.at(2, 3)), int &>::value, "");
    auto i = ttl::idx(2, 3);
    ASSERT_EQ(x[i], 23);
    static_assert(std::is_same<decltype(x[i]), int &>::value, "");
    x[i] += 100;
    ASSERT_EQ(x[i], 123);

    auto vx = ttl::view(x);
    static_assert(std::is_same<decltype(vx[i]), const int &>::value, "");
}
