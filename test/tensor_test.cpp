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

int main()
{
    test_1();
    return 0;
}
