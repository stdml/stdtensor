#include "../testing.hpp"

#include <stdml/bits/tensor/cudart.hpp>

TEST(libtensor_test, test_detect)
{
    auto ok = stdml::has_cuda();
    printf("%d\n", ok);
    int n = stdml::get_cuda_gpu_count();
    if (!ok) {
        ASSERT_EQ(n, 0);
    } else {
        // FIXME: maybe == 0
        ASSERT_TRUE(n > 0);
    }
}
