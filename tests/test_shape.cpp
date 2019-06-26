#include "testing.hpp"

#include <ttl/tensor>

using dim_t = uint32_t;

template <ttl::internal::rank_t r>
using shape = ttl::internal::basic_shape<r, dim_t>;

void test_shape(dim_t h, dim_t w)
{
    shape<2> s(h, w);
    ASSERT_EQ(s.size(), h * w);

    dim_t k = 0;
    for (dim_t i = 0; i < h; ++i) {
        for (dim_t j = 0; j < w; ++j) {
            ASSERT_EQ(s.offset(i, j), k);
            ++k;
        }
    }
}

TEST(shape_test, test1)
{
    for (dim_t h = 1; h < 10; ++h) {
        for (dim_t w = 1; w < 10; ++w) { test_shape(h, w); }
    }

    {
        shape<5> s(10, 10, 10, 10, 10);
        ASSERT_EQ(s.offset(1, 2, 3, 4, 5), static_cast<dim_t>(12345));
    }
}

TEST(shape_test, assign_test)
{
    shape<3> t(2, 3, 4);
    shape<3> s = t;
    ASSERT_EQ(s.size(), static_cast<dim_t>(24));
}

TEST(shape_test, make_test)
{
    {
        const auto s = ttl::make_shape();
        ASSERT_EQ(s.size(), static_cast<dim_t>(1));
        ASSERT_EQ(s, ttl::shape<0>());
    }
    {
        const auto s = ttl::make_shape(2);
        ASSERT_EQ(s.size(), static_cast<dim_t>(2));
        ASSERT_EQ(s, ttl::shape<1>(2));
    }
    {
        const auto s = ttl::make_shape(2, 3);
        ASSERT_EQ(s.size(), static_cast<dim_t>(6));
        ASSERT_EQ(s, ttl::shape<2>(2, 3));
    }
    {
        const auto s = ttl::make_shape(2, 3, 4);
        ASSERT_EQ(s.size(), static_cast<dim_t>(24));
        ASSERT_EQ(s, ttl::shape<3>(2, 3, 4));
    }
}

TEST(shape_test, cmp_test)
{
    ASSERT_TRUE(ttl::shape<2>(2, 3) != ttl::shape<2>(3, 2));
    ASSERT_FALSE(ttl::shape<2>(2, 3) != ttl::shape<2>(2, 3));
}
