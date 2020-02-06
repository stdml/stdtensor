#include "testing.hpp"

#include <ttl/tensor>

TEST(scalar_test, test_constructor)
{
    {
        ttl::tensor<int, 0> x;
        ttl::tensor_ref<int, 0> r(x);
        ttl::tensor_view<int, 0> v(x);
    }
    {
        int value = 0;
        ttl::tensor_ref<int, 0> r(&value);
        ttl::tensor_view<int, 0> v(&value);
    }
}
