#include "../testing.hpp"

#include <stdml/tensor>
#include <ttl/experimental/flat_tensor>

void call_with_view(const stdml::TensorView &x) {}

TEST(libtensor_test, test_construct)
{
    namespace ml = stdml;
    ml::Tensor x1(ml::i32);
    // ml::Tensor x2(ml::i32, {});
    ml::Tensor x3(ml::i32, {1});
    ml::Tensor x4(ml::i32, {2, 3});

    ttl::tensor<int32_t, 2> y(3, 4);
    ml::Tensor x5(std::move(y));
    ASSERT_EQ(y.data(), nullptr);

    size_t bs = 100;
    ml::Tensor x6(ml::i32, {(int)bs, 2, 3});

    ml::TensorRef x7 = x6.ref();
    call_with_view(x7);
    call_with_view(x7[0]);
    ml::TensorView x8 = x7.view();
}

TEST(libtensor_test, test_1)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ml::Shape());
    ml::Tensor y = ml::Tensor::new_like(x);

    {
        auto r = x.ref();
        static_assert(std::is_same<ml::TensorRef, decltype(r)>::value, "");
    }
    {
        auto v = x.view();
        static_assert(std::is_same<ml::TensorView, decltype(v)>::value, "");
    }
}

TEST(libtensor_test, test_typed)
{
    namespace ml = stdml;
    ml::Tensor x(ml::f32, ttl::make_shape(2, 3, 4));
    auto y = x.typed<float>();
    static_assert(std::is_same<ttl::experimental::flat_tensor_ref<float>,
                               decltype(y)>::value,
                  "");
}

TEST(libtensor_test, test_flatten)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ttl::make_shape(2, 3, 4));
    auto y = x.flatten<int32_t>();
    static_assert(std::is_same<ttl::tensor_ref<int32_t, 1>, decltype(y)>::value,
                  "");
}

TEST(libtensor_test, test_data_end)
{
    namespace ml = stdml;
    using T = int32_t;
    ml::Tensor x(ml::i32, ttl::make_shape(2, 3, 4));
    auto r = x.ref();
    auto v = x.view();

    x.data<T>();
    x.data_end<T>();

    r.data<T>();
    r.data_end<T>();

    v.data<T>();
    v.data_end<T>();
}

TEST(libtensor_test, test_slice)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ml::shape(10));
    std::iota(x.data<int32_t>(), x.data_end<int32_t>(), 0);
    for (int i = 0; i < 10; ++i) {
        int xi = x[i].data<int32_t>()[0];
        ASSERT_EQ(xi, i);
    }
    auto y = x.slice(5, 10);
    int s = std::accumulate(y.data<int32_t>(), y.data_end<int32_t>(), 0);
    ASSERT_EQ(s, 35);
    {
        int tot = 0;
        for (int i = 0; i < 5; ++i) {
            int yi = y.view()[i].data<int32_t>()[0];
            tot += yi;
        }
        ASSERT_EQ(tot, 35);
    }
}
