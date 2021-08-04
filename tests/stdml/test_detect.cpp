#include "../testing.hpp"

#include <stdml/bits/tensor/allocator.hpp>

TEST(libtensor_test, test_detect)
{
    auto ok = stdml::has_cuda();
    printf("%d\n", ok);
}
