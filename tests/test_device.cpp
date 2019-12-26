#include "testing.hpp"

#include <ttl/device>

template <typename D>
void f()
{
}

TEST(device_test, test_device)
{
    f<ttl::host_memory>();
    f<ttl::cuda_memory>();
}
