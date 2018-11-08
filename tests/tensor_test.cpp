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

auto create_tensor_func()
{
    tensor<int, 5> t(3, 4, 5, 6, 7);
    return t;
}

TEST(tensor_test, return_tensor) { auto t = create_tensor_func(); }

template <typename R> R read_tensor_func(const tensor<R, 2> &t, int i, int j)
{
    const R x = t.at(i, j);
    return x;
}

template <typename R>
R read_tensor_ref_func(const tensor_ref<R, 2> &t, int i, int j)
{
    const R x = t.at(i, j);
    return x;
}

template <typename R>
R read_tensor_view_func(const tensor_view<R, 2> &t, int i, int j)
{
    const R x = t.at(i, j);
    return x;
}

TEST(tensor_test, test_read_access)
{
    tensor<int, 2> t(2, 2);

    scalar(t[0][0]) = 1;
    ASSERT_EQ(1, read_tensor_func(t, 0, 0));

    scalar(t[0][0]) = 2;
    ASSERT_EQ(2, read_tensor_ref_func(ref(t), 0, 0));

    scalar(t[0][0]) = 3;
    ASSERT_EQ(3, read_tensor_view_func(view(t), 0, 0));

    {
        auto v = view(t);
        auto p = v.at(0, 0);
        p += 1;
        ASSERT_EQ(3, read_tensor_view_func(view(t), 0, 0));
    }
    {
        auto &p = t.at(0, 0);
        p += 1;
        ASSERT_EQ(4, read_tensor_view_func(view(t), 0, 0));
    }
}

TEST(tensor_test, test_scalar_assignment)
{
    tensor<int, 2> t(2, 2);
    int x = 0;

    x = t[0][0] = 1;
    ASSERT_EQ(1, read_tensor_func(t, 0, 0));
    ASSERT_EQ(1, x);

    x = t[0][0] = 2;
    ASSERT_EQ(2, read_tensor_ref_func(ref(t), 0, 0));

    t[0][0] = 3;
    ASSERT_EQ(3, read_tensor_view_func(view(t), 0, 0));

    tensor<int, 0> s;
    s = 1;
    ASSERT_EQ(1, s.data()[0]);
    x = s = 2;
    ASSERT_EQ(2, s.data()[0]);
    ASSERT_EQ(2, x);
}

template <template <typename, ttl::internal::rank_t, typename> class T,
          typename R, ttl::internal::rank_t r, typename shape_type>
void test_static_properties(const T<R, r, shape_type> &x)
{
    using t = T<R, r, shape_type>;
    static_assert(std::is_same<typename t::value_type, R>::value,
                  "invalid value_type");
    static_assert(t::rank == r, "invalid rank");
    static_assert(decltype(x.shape())::rank == r, "invalid rank of shape");
    static_assert(
        std::is_same<typename std::remove_const<typename std::remove_pointer<
                         decltype(x.data())>::type>::type,
                     R>::value,
        "invalid data type");
}

TEST(tensor_test, test_static_properties)
{
    {
        tensor<float, 0> t;
        test_static_properties(t);
        test_static_properties(ref(t));
        test_static_properties(view(t));
    }
    {
        tensor<float, 1> t(1);
        test_static_properties(t);
        test_static_properties(ref(t));
        test_static_properties(view(t));
    }
    {
        tensor<float, 2> t(1, 1);
        test_static_properties(t);
        test_static_properties(ref(t));
        test_static_properties(view(t));
    }
    {
        tensor<float, 3> t(1, 1, 1);
        test_static_properties(t);
        test_static_properties(ref(t));
        test_static_properties(view(t));
    }
}
