#include "testing.hpp"

#include <ttl/algorithm>
#include <ttl/cuda_tensor>
#include <ttl/device>
#include <ttl/experimental/copy>
#include <ttl/range>
#include <ttl/tensor>

void test_copy(int n)
{
    ttl::tensor<int, 1> x_host(n);
    ttl::cuda_tensor<int, 1> x_cuda(n);

    ttl::fill(ttl::ref(x_host), 1);
    ttl::copy(ttl::ref(x_cuda), ttl::view(x_host));

    ttl::fill(ttl::ref(x_host), 2);
    for (auto i : ttl::range<0>(x_host)) { ASSERT_EQ(x_host.data()[i], 2); }

    ttl::copy(ttl::ref(x_host), ttl::view(x_cuda));
    for (auto i : ttl::range<0>(x_host)) { ASSERT_EQ(x_host.data()[i], 1); }
}

TEST(copy_test, test_copy)
{
    test_copy(1);
    test_copy(2);
    test_copy(10);
    test_copy(100);
    test_copy(1000);
}
