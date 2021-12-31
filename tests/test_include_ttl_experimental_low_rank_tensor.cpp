#include "testing.hpp"

#include <ttl/experimental/low_rank_tensor>

template <typename R, int n>
void test_vec()
{
    using ttl::vec;
    static_assert(sizeof(vec<R, n>) == sizeof(R) * n, "");
}

template <typename R, int m, int n = m>
void test_mat()
{
    using ttl::mat;
    static_assert(sizeof(mat<R, m, n>) == sizeof(R) * m * n, "");
}

TEST(ttl_low_rank_tensor_include_test, test_vec)
{
    // test_vec<int, 0>();
    test_vec<int, 1>();
    test_vec<int, 2>();
    test_vec<int, 3>();
}

TEST(ttl_low_rank_tensor_include_test, test_mat)
{
    // test_mat<int, 0>();
    test_mat<int, 1>();
    test_mat<int, 2>();
    test_mat<int, 3>();

    test_mat<int, 1, 2>();
    test_mat<int, 1, 3>();
    test_mat<int, 1, 4>();

    test_mat<int, 2, 3>();
    test_mat<int, 2, 4>();
    test_mat<int, 3, 4>();
}

TEST(ttl_low_rank_tensor_include_test, test_op)
{
    using ttl::vec;
    vec<int, 2> x;
    {
        vec<int, 2> y = -x;
        static_assert(sizeof(y) > 0, "");
    }
    {
        vec<int, 2> y = x * 2;
        static_assert(sizeof(y) > 0, "");
    }
}
