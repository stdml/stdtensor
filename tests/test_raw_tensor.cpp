#include "testing.hpp"

#include <ttl/tensor>

using ttl::experimental::raw_tensor;

TEST(raw_tensor_test, test1)
{
    using encoder = raw_tensor::encoder_type;
    using raw_shape = raw_tensor::shape_type;

    {
        raw_tensor t(encoder::value<float>());
        ASSERT_EQ(t.shape().size(), static_cast<raw_shape::dimension_type>(1));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        t.ref_as<float, 0>();
        t.view_as<float, 0>();
    }
    {
        raw_tensor t(encoder::value<float>(), 1);
        ASSERT_EQ(t.shape().size(), static_cast<raw_shape::dimension_type>(1));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        t.ref_as<float, 1>();
        t.view_as<float, 1>();
    }
    {
        raw_tensor t(encoder::value<float>(), 1, 2);
        ASSERT_EQ(t.shape().size(), static_cast<raw_shape::dimension_type>(2));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        t.ref_as<float, 2>();
        t.view_as<float, 2>();
    }
    {
        raw_tensor t(encoder::value<float>(), 1, 2, 3);
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.shape().size(), static_cast<raw_shape::dimension_type>(6));
        ASSERT_EQ(t.data<float>(), t.data());
        t.ref_as<float, 3>();
        t.view_as<float, 3>();
    }
}
