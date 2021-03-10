#include "../testing.hpp"

#include <stdml/tensor>

TEST(libtensor_test, test_show)
{
    namespace ml = stdml;
    {
        ml::Tensor x(ml::i32, ml::Shape());
        const auto s = ml::show(x);
        ASSERT_EQ(s, std::string("i32()0"));
    }  // namespace ml=stdml;
    {
        ml::Tensor x(ml::i32, ttl::make_shape(2, 3));
        std::iota(x.data<int32_t>(), x.data<int32_t>() + x.size(), 0);
        const auto s = ml::show(x);
        ASSERT_EQ(s, std::string("i32(2, 3)[[0, 1, 2], [3, 4, 5]]"));
    }
}
