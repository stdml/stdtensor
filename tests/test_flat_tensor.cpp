#include "testing.hpp"

#include <ttl/experimental/flat_tensor>
#include <ttl/tensor>

TEST(flat_tensor_test, test1)
{
    using ttl::experimental::flat_tensor;
    {
        flat_tensor<float> t;
        using flat_shape = flat_tensor<float>::shape_type;

        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(1));
        {
            auto x = t.ranked<0>();
            static_assert(
                std::is_same<decltype(x), ttl::tensor_ref<float, 0>>::value,
                "");
        }
        t.data();
        static_assert(std::is_same<decltype(t)::value_type, float>::value, "");

        {
            const auto r = ttl::ref(t);
            auto x = r.ranked<0>();
            static_assert(
                std::is_same<decltype(x), ttl::tensor_ref<float, 0>>::value,
                "");
        }
        {
            const auto v = ttl::view(t);
            auto x = v.ranked<0>();
            static_assert(
                std::is_same<decltype(x), ttl::tensor_view<float, 0>>::value,
                "");
        }
    }
    {
        flat_tensor<float> t(1);
        using flat_shape = flat_tensor<float>::shape_type;
        ASSERT_EQ(t.size(), static_cast<flat_shape::dimension_type>(1));
        t.ranked<1>();
        t.data();
    }
    {
        flat_tensor<float> t(1, 2);
        using flat_shape = flat_tensor<float>::shape_type;
        ASSERT_EQ(t.size(), static_cast<flat_shape::dimension_type>(2));
        t.ranked<2>();
        t.data();
    }
    {
        flat_tensor<float> t(1, 2, 3);
        using flat_shape = flat_tensor<float>::shape_type;
        ASSERT_EQ(t.size(), static_cast<flat_shape::dimension_type>(6));
        t.ranked<3>();
        t.data();
    }
}
