#include "testing.hpp"

#include <ttl/algorithm>
#include <ttl/tensor>

TEST(tensor_algo_test, test_argmax)
{
    using R = float;
    ttl::tensor<R, 1> t(10);
    std::iota(t.data(), t.data_end(), 1);
    ASSERT_EQ(static_cast<uint32_t>(9), ttl::argmax(ttl::view(t)));
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
    ttl::cast(ttl::view(x), ttl::ref(y));

    ASSERT_EQ(5, ttl::sum(ttl::view(y)));
}

TEST(tensor_algo_test, test_fill)
{
    {
        using R = int;
        ttl::tensor<R, 1> t(10);
        ttl::fill(ttl::ref(t), 1);
    }
    {
        using R = float;
        ttl::tensor<R, 1> t(10);
        ttl::fill(ttl::ref(t), static_cast<R>(1.1));
    }
}

TEST(tensor_algo_test, test_hamming_distance)
{
    using R = int;
    int n = 0xffff;
    ttl::tensor<R, 1> x(n);
    ttl::fill(ttl::ref(x), -1);
    ttl::tensor<R, 1> y(n);
    ttl::fill(ttl::ref(y), 1);
    ASSERT_EQ(static_cast<uint32_t>(n),
              ttl::hamming_distance(ttl::view(x), ttl::view(y)));
}

TEST(tensor_algo_test, chebyshev_distenace)
{
    using R = int;
    int n = 0xffff;
    ttl::tensor<R, 1> x(n);
    ttl::tensor<R, 1> y(n);
    std::iota(x.data(), x.data_end(), 1);
    std::iota(y.data(), y.data_end(), 1);
    ASSERT_EQ(static_cast<R>(0),
              ttl::chebyshev_distenace(ttl::view(x), ttl::view(y)));
    std::reverse(y.data(), y.data_end());
    ASSERT_EQ(static_cast<R>(n - 1),
              ttl::chebyshev_distenace(ttl::view(x), ttl::view(y)));
}

TEST(tensor_algo_test, test_summaries_int)
{
    using R = int;
    const int n = 10;
    ttl::tensor<R, 1> x(n);
    std::iota(x.data(), x.data_end(), -5);
    ASSERT_EQ(-5, ttl::min(ttl::view(x)));
    ASSERT_EQ(4, ttl::max(ttl::view(x)));
    ASSERT_EQ(-5, ttl::sum(ttl::view(x)));
    ASSERT_EQ(0, ttl::mean(ttl::view(x)));
}

TEST(tensor_algo_test, test_summaries_float)
{
    using R = float;
    const int n = 10;
    ttl::tensor<R, 1> x(n);
    std::iota(x.data(), x.data_end(), -5);
    ASSERT_EQ(-5, ttl::min(ttl::view(x)));
    ASSERT_EQ(4, ttl::max(ttl::view(x)));
    ASSERT_EQ(-5, ttl::sum(ttl::view(x)));
    ASSERT_EQ(-0.5, ttl::mean(ttl::view(x)));
}
