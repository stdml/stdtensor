#include "testing.hpp"

#include <ttl/shape>

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
            {
                dim_t u, v;
                std::tie(u, v) = s.expand(k);
                ASSERT_EQ(i, u);
                ASSERT_EQ(j, v);
            }
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

TEST(shape_test, accessors_test)
{
    ttl::shape<3> shape(2, 3, 4);
    auto dims = shape.dims();

    using dim_t = ttl::shape<3>::dimension_type;
    ASSERT_EQ(dims[0], static_cast<dim_t>(2));
    ASSERT_EQ(dims[1], static_cast<dim_t>(3));
    ASSERT_EQ(dims[2], static_cast<dim_t>(4));
}

void test_coord_3(dim_t l, dim_t m, dim_t n)
{
    ttl::shape<3> s(l, m, n);

    for (dim_t i = 0; i < l; ++i) {
        for (dim_t j = 0; j < m; ++j) {
            for (dim_t k = 0; k < n; ++k) {
                const dim_t idx = s.offset(i, j, k);
                ASSERT_EQ(i, s.coord<0>(idx));
                ASSERT_EQ(j, s.coord<1>(idx));
                ASSERT_EQ(k, s.coord<2>(idx));
            }
        }
    }
}

TEST(shape_test, test_coord)
{
    for_all_permutations(test_coord_3, 2, 2, 2);
    for_all_permutations(test_coord_3, 3, 3, 3);
    for_all_permutations(test_coord_3, 2, 3, 4);
    for_all_permutations(test_coord_3, 2, 4, 8);
}

TEST(shape_test, test_join)
{
    using ttl::make_shape;
    using ttl::internal::join_shape;

    const auto s0 = ttl::make_shape(1, 2, 3, 4);
    ASSERT_EQ(s0, join_shape(make_shape(1, 2), make_shape(3, 4)));
    ASSERT_EQ(s0, join_shape(make_shape(1, 2, 3), make_shape(4)));
    ASSERT_EQ(s0, join_shape(make_shape(1, 2, 3, 4), make_shape()));
    ASSERT_EQ(s0, join_shape(make_shape(), make_shape(1, 2, 3, 4)));

    ASSERT_EQ(make_shape(), join_shape(make_shape(), make_shape()));
}

TEST(shape_test, test_batch)
{
    using ttl::make_shape;
    using ttl::internal::batch;

    ASSERT_EQ(make_shape(1), batch(1, make_shape()));
    ASSERT_EQ(make_shape(2, 3), batch(2, make_shape(3)));
}

TEST(shape_test, test_flatten)
{
    using ttl::internal::flatten_shape;
    const ttl::shape<6> s(1, 2, 3, 4, 5, 6);

    const auto s0 = flatten_shape<0, 6>()(s);
    ASSERT_EQ(s0, ttl::shape<2>(1, 720));

    const auto s1 = flatten_shape<1, 5>()(s);
    ASSERT_EQ(s1, ttl::shape<2>(1, 720));

    const auto s2 = flatten_shape<2, 4>()(s);
    ASSERT_EQ(s2, ttl::shape<2>(2, 360));

    const auto s3 = flatten_shape<3, 3>()(s);
    ASSERT_EQ(s3, ttl::shape<2>(6, 120));

    const auto s4 = flatten_shape<4, 2>()(s);
    ASSERT_EQ(s4, ttl::shape<2>(24, 30));

    const auto s5 = flatten_shape<5, 1>()(s);
    ASSERT_EQ(s5, ttl::shape<2>(120, 6));

    const auto s6 = flatten_shape<6, 0>()(s);
    ASSERT_EQ(s6, ttl::shape<2>(720, 1));

    const auto s7 = flatten_shape<6>()(s);
    ASSERT_EQ(s7, ttl::shape<1>(720));

    const auto s8 = flatten_shape<>()(s);
    ASSERT_EQ(s8, ttl::shape<1>(720));
}
