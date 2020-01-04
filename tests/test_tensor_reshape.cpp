#include "testing.hpp"

#include <type_traits>

#include <ttl/tensor>

TEST(tensor_reshape_test, test_chunk)
{
    int n = 12;
    ttl::tensor<int, 2> x(n, 5);
    auto y = ttl::chunk(x, 3);
    auto z = ttl::flatten(ttl::view(y));

    static_assert(std::is_same<decltype(y), ttl::tensor_ref<int, 3>>::value,
                  "");
    static_assert(std::is_same<decltype(z), ttl::tensor_view<int, 1>>::value,
                  "");
}

template <typename R, ttl::rank_t r>
void test_flatten(const ttl::tensor<R, r> &t)
{
    {
        auto v = ttl::flatten(t);
        static_assert(std::is_same<decltype(v), ttl::vector_ref<R>>::value, "");
        ASSERT_EQ(v.shape().size(), t.shape().size());
    }
    {
        auto v = ttl::flatten(ttl::ref(t));
        static_assert(std::is_same<decltype(v), ttl::vector_ref<R>>::value, "");
        ASSERT_EQ(v.shape().size(), t.shape().size());
    }
    {
        auto v = ttl::flatten(ttl::view(t));
        static_assert(std::is_same<decltype(v), ttl::vector_view<R>>::value,
                      "");
        ASSERT_EQ(v.shape().size(), t.shape().size());
    }
}

TEST(tensor_reshape_test, test_flatten)
{
    {
        ttl::tensor<int, 0> x;
        test_flatten(x);
    }
    {
        ttl::tensor<int, 1> x(4);
        test_flatten(x);
    }
    {
        ttl::tensor<int, 2> x(4, 5);
        test_flatten(x);
    }
}
