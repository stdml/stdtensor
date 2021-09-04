#include "testing.hpp"

#include <ttl/functional>
#include <ttl/tensor>

TEST(functional_test, test_1)
{
    ttl::tensor<int, 1> x(10);
    std::iota(x.data(), x.data_end(), 1);
    // ttl::length<>()(x);
    auto y = ttl::slice<>(5, 10)(x);
    ASSERT_EQ(y[0], 6);
    ASSERT_EQ(y[2], 8);
    ASSERT_EQ(y[4], 10);
}
