#include <gtest/gtest.h>

#include <stdtensor>

using namespace ttl::internal;

using dim_t = uint32_t;
template <rank_t r> using shape = basic_shape<r, dim_t>;

void test_shape(int h, int w)
{
    shape<2> s(h, w);
    ASSERT_EQ(s.size(), h * w);

    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            ASSERT_EQ(s.offset(i, j), k);
            ++k;
        }
    }
}

TEST(shape_test, test1)
{
    for (int h = 1; h < 10; ++h) {
        for (int w = 1; w < 10; ++w) { test_shape(h, w); }
    }

    {
        shape<5> s(10, 10, 10, 10, 10);
        ASSERT_EQ(s.offset(1, 2, 3, 4, 5), 12345);
    }
}
