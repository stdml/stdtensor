#include <stdtensor>

void example_1()
{
    using pixel_t = std::array<uint8_t, 3>;
    static_assert(sizeof(pixel_t) == 3, "invalid pixel size");
    using bmp_t = matrix<pixel_t>;

    int w = 1024;
    int h = 768;
    bmp_t img(h, w);  // Note that img is fully packed, without row padding
}

void example_2()
{
    const int h = 3;
    const int w = 4;
    const int c = 5;
    const int n = h * w * c;

    tensor<int, 3> t(h, w, c);
    {
        int idx = 0;
        for (const auto &t1 : t) {
            for (const auto &t2 : t1) {
                for (const auto &t3 : t2) { scalar(t3) = idx++; }
            }
        }
    }
    {
        tensor_ref<int, 3> r(t.data(), t.shape());
        printf("%d\n", r.shape().offset(2, 3, 4));
        scalar(r[2][3][4]) = 0;
    }

    {
        tensor_view<int, 3> v(t.data(), t.shape());
        int tot = 0;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                for (int k = 0; k < c; ++k) { tot += scalar(v[i][j][k]); }
            }
        }

        printf("%d, %d\n", n * (n + 1) / 2, tot);
    }
}

int main()
{
    example_1();
    example_2();
    return 0;
}
