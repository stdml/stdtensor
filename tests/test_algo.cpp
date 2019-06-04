#include "testing.hpp"

#include <ttl/algorithm>
#include <ttl/tensor>

TEST(tensor_algo_test, test_argmax)
{
    using R = float;
    ttl::tensor<R, 1> t(10);
    std::iota(t.data(), t.data_end(), 1);
    ASSERT_EQ(9, ttl::argmax(view(t)));
}

TEST(tensor_algo_test, test_cast)
{
    int n = 10;

    ttl::tensor<float, 1> x(n);
    std::generate(x.data(), x.data_end(), [v = 0.1]() mutable {
        auto u = v;
        v += 0.2;
        return u;
    });

    ttl::tensor<int, 1> y(n);
    ttl::cast(view(x), ref(y));

    ASSERT_EQ(5, ttl::sum(view(y)));
}

TEST(tensor_algo_test, test_fill)
{
    {
        using R = int;
        ttl::tensor<R, 1> t(10);
        ttl::fill(ref(t), 1);
    }
    {
        using R = float;
        ttl::tensor<R, 1> t(10);
        ttl::fill(ref(t), static_cast<R>(1.1));
    }
}

TEST(tensor_algo_test, test_hamming_distance)
{
    using R = int;
    int n = 0xffff;
    ttl::tensor<R, 1> x(n);
    ttl::fill(ref(x), -1);
    ttl::tensor<R, 1> y(n);
    ttl::fill(ref(y), 1);
    ASSERT_EQ(n, ttl::hamming_distance(view(x), view(y)));
}
