#include "testing.hpp"

#include <ttl/tensor>

using ttl::experimental::flat_tensor;

TEST(flat_tensor_test, test1)
{
    {
        flat_tensor<float> t;
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<0>();
        t.view_as<0>();

        static_assert(std::is_same<decltype(t)::value_type, float>::value, "");
    }
    {
        flat_tensor<float> t(1);
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<1>();
        t.view_as<1>();
    }
    {
        flat_tensor<float> t(1, 2);
        ASSERT_EQ(t.shape().size(), 2);
        t.ref_as<2>();
        t.view_as<2>();
    }
    {
        flat_tensor<float> t(1, 2, 3);
        ASSERT_EQ(t.shape().size(), 6);
        t.ref_as<3>();
        t.view_as<3>();
    }
}
