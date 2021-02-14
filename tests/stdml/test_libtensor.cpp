#include "../testing.hpp"

#include <stdml/tensor.hpp>

TEST(libtensor_test, test_1)
{
    namespace ml = stdml;
    ml::Tensor x(ml::i32, ml::Shape());

    {
        auto r = x.ref();
        static_assert(std::is_same<ml::TensorRef, decltype(r)>::value, "");
    }
    {
        auto v = x.view();
        static_assert(std::is_same<ml::TensorView, decltype(v)>::value, "");
    }
}
