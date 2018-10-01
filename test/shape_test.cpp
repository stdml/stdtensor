#include <cassert>

#include <stdtensor>

void test_1()
{
    int h = 2;
    int w = 3;
    shape<2> s(h, w);
    assert(s.size() == h * w);

    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) { assert(s.index(i, j) == k++); }
    }
}

int main()
{
    test_1();
    return 0;
}
