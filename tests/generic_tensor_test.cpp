#include <gtest/gtest.h>

#include <stdtensor>

using ttl::generic_shape;
using ttl::generic_tensor;

TEST(generic_tensor_test, test1)
{
    {
        generic_tensor t(ttl::internal::typeinfo<float>());
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<float, 0>();
        t.view_as<float, 0>();
    }
    {
        generic_tensor t(ttl::internal::typeinfo<float>(), 1);
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<float, 1>();
        t.view_as<float, 1>();
    }
    {
        generic_tensor t(ttl::internal::typeinfo<float>(), 1, 2);
        ASSERT_EQ(t.shape().size(), 2);
        t.ref_as<float, 2>();
        t.view_as<float, 2>();
    }
    {
        generic_tensor t(ttl::internal::typeinfo<float>(), 1, 2, 3);
        ASSERT_EQ(t.shape().size(), 6);
        t.ref_as<float, 3>();
        t.view_as<float, 3>();
    }
}
