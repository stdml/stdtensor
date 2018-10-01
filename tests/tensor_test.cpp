#include <cassert>
#include <numeric>

#include <stdtensor>

void test_1()
{
    int h = 2;
    int w = 3;
    tensor<int, 2> t(h, w);
    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) { t.at(i, j) = k++; }
    }

    int sum = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) { sum += t.at(i, j); }
    }

    int n = h * w - 1;
    assert(sum == n * (n + 1) / 2);
}

void test_2()
{
    using pixel_t = std::array<uint8_t, 3>;
    static_assert(sizeof(pixel_t) == 3, "invalid pixel size");
    using bmp_t = matrix<pixel_t>;

    int h = 1024;
    int w = 768;
    bmp_t img(h, w);  // Note that img is fully packed, without row padding
}

template <bool write, typename T> struct test_assign_ {
    void operator()(T &x, int v) { x = v; }
};

template <typename T> struct test_assign_<false, T> {
    void operator()(T &x, int v) { x = v; }
};

template <bool write, typename T> void test_assign(T &&x, int v)
{
    test_assign_<write, T>()(x, v);
}

template <typename T, bool write = true> struct test_5d_array {
    void operator()(const T &t)
    {
        t[1];
        t[1][2];
        t[1][2][3];
        t[1][2][3][4];
        t[1][2][3][4][5];

        {
            int idx = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 5; ++k) {
                        for (int l = 0; l < 6; ++l) {
                            for (int m = 0; m < 7; ++m) {
                                test_assign<write>(scalar(t[i][j][k][l][m]),
                                                   ++idx);
                            }
                        }
                    }
                }
            }
            int n = 3 * 4 * 5 * 6 * 7;
            int tot = std::accumulate(t.data(), t.data() + n, 0);
            if (write) { assert(n * (n + 1) / 2 == tot); }
        }
    }
};

void test_3()
{
    tensor<int, 5> t(3, 4, 5, 6, 7);
    tensor_ref<int, 5> r(t.data(), t.shape());
    tensor_view<int, 5> v(t.data(), t.shape());

    test_5d_array<decltype(t)>()(t);
    test_5d_array<decltype(r)>()(r);
    test_5d_array<decltype(v), false>()(v);
}

int main()
{
    test_1();
    test_2();
    test_3();
    return 0;
}
