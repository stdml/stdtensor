#include "../testing.hpp"

#include <stdml/tensor.hpp>

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

TEST(libtensor_test, test_flatten)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ttl::make_shape(2, 3, 4));
    auto y = x.flatten<int32_t>();
    static_assert(std::is_same<ttl::tensor_ref<int32_t, 1>, decltype(y)>::value,
                  "");
}

TEST(libtensor_test, test_show)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ml::Shape());
}
