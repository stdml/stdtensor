#include <cassert>

#include <stdtensor>

using dim_t = uint32_t;
template <rank_t r> using shape = basic_shape<r, dim_t>;

void test_shape(int h, int w)
{
    shape<2> s(h, w);
    assert(s.size() == h * w);

    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) { assert(s.offset(i, j) == k++); }
    }
}

void test_1()
{
    for (int h = 1; h < 10; ++h) {
        for (int w = 1; w < 10; ++w) { test_shape(h, w); }
    }

    {
        shape<5> s(10, 10, 10, 10, 10);
        assert(s.offset(1, 2, 3, 4, 5) == 12345);
    }
}

int main()
{
    test_1();
    return 0;
}
