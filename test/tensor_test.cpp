#include <cassert>

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

int main()
{
    test_1();
    test_2();
    return 0;
}
