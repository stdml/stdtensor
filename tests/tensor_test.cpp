#include <gtest/gtest.h>

#include <numeric>
#include <type_traits>

#include <stdtensor>

using namespace ttl;

TEST(tensor_test, test1)
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
    ASSERT_EQ(sum, n * (n + 1) / 2);
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
            if (write) { ASSERT_EQ(n * (n + 1) / 2, tot); }
        }

        if (write) {
            int idx = 0;
            for (const auto &t1 : t) {
                for (const auto &t2 : t1) {
                    for (const auto &t3 : t2) {
                        for (const auto &&t4 : t3) {
                            for (auto &&t5 : t4) {
                                ++idx;
                                auto v = scalar(t5);
                                ASSERT_EQ(v, idx);
                            }
                        }
                    }
                }
            }
        }
    }
};

TEST(tensor_test, test3)
{
    tensor<int, 5> t(3, 4, 5, 6, 7);
    tensor_ref<int, 5> r(t.data(), t.shape());
    tensor_view<int, 5> v(t.data(), t.shape());

    test_5d_array<decltype(t)>()(t);
    test_5d_array<decltype(r)>()(r);
    test_5d_array<decltype(v), false>()(v);
}

template <typename T, uint8_t r> void ref_func(const tensor_ref<T, r> &x) {}

template <typename T, uint8_t r> void test_auto_ref()
{
    static_assert(std::is_convertible<tensor<T, r>, tensor_ref<T, r>>::value,
                  "can't convert to ref");
}

TEST(tensor_test, auto_ref)
{
    test_auto_ref<int, 0>();
    test_auto_ref<int, 1>();
    test_auto_ref<int, 2>();

    tensor<int, 5> t(3, 4, 5, 6, 7);
    ref_func(ref(t));

    tensor_ref<int, 5> r = t;
    ref_func(r);

    ref_func(tensor_ref<int, 5>(t));
    // f(t);  // NOT possible
}

template <typename T, uint8_t r> void view_func(const tensor_view<T, r> &x) {}

template <typename T, uint8_t r> void test_auto_view()
{
    static_assert(std::is_convertible<tensor<T, r>, tensor_view<T, r>>::value,
                  "can't convert to ref");
    static_assert(
        std::is_convertible<tensor_ref<T, r>, tensor_view<T, r>>::value,
        "can't convert to ref");
}

TEST(tensor_test, auto_view)
{
    test_auto_view<int, 0>();
    test_auto_view<int, 1>();
    test_auto_view<int, 2>();

    tensor<int, 5> t(3, 4, 5, 6, 7);
    view_func(view(t));

    tensor_view<int, 5> r = t;
    view_func(r);

    view_func(tensor_view<int, 5>(t));
    // view_func(t);  // NOT possible
}

auto f2()
{
    tensor<int, 5> t(3, 4, 5, 6, 7);
    return t;
}

TEST(tensor_test, return_tensor) { auto t = f2(); }
