#include "testing.hpp"

#include <ttl/tensor>

using ttl::experimental::flat_tensor;

TEST(flat_tensor_test, test1)
{
    {
        flat_tensor<float> t;
        using flat_shape = flat_tensor<float>::shape_type;

        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(1));
        t.ref_as<0>();
        t.view_as<0>();
        t.data();
        static_assert(std::is_same<decltype(t)::value_type, float>::value, "");

        const auto r = ref(t);
        r.ref_as<0>();
        r.view_as<0>();
    }
    {
        flat_tensor<float> t(1);
        using flat_shape = flat_tensor<float>::shape_type;

        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(1));
        t.ref_as<1>();
        t.view_as<1>();
        t.data();
    }
    {
        flat_tensor<float> t(1, 2);
        using flat_shape = flat_tensor<float>::shape_type;

        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(2));
        t.ref_as<2>();
        t.view_as<2>();
        t.data();
    }
    {
        flat_tensor<float> t(1, 2, 3);
        using flat_shape = flat_tensor<float>::shape_type;

        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(6));
        t.ref_as<3>();
        t.view_as<3>();
        t.data();
    }
}
