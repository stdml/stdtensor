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
