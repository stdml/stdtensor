#include "testing.hpp"

#include "fake_cuda_runtime.h"

#include <ttl/bits/std_cuda_tensor.hpp>
#include <ttl/tensor>

using ttl::tensor;
using ttl::internal::cuda_tensor;

TEST(cuda_tensor_test, test1)
{
    using R = float;
    cuda_tensor<R, 2> m1(10, 100);
}

TEST(cuda_tensor_test, test2)
{
    using R = float;
    cuda_tensor<R, 2> m1(10, 100);
    tensor<R, 2> m2(10, 100);

    m1.fromHost(m2.data());
    m1.toHost(m2.data());
}
