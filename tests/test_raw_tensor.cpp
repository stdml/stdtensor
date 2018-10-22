#include <gtest/gtest.h>

#include <stdtensor>

using ttl::raw_shape;
using ttl::raw_tensor;

TEST(raw_tensor_test, test1)
{
    {
        raw_tensor t(ttl::internal::typeinfo<float>());
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<float, 0>();
        t.view_as<float, 0>();
    }
    {
        raw_tensor t(ttl::internal::typeinfo<float>(), 1);
        ASSERT_EQ(t.shape().size(), 1);
        t.ref_as<float, 1>();
        t.view_as<float, 1>();
    }
    {
        raw_tensor t(ttl::internal::typeinfo<float>(), 1, 2);
        ASSERT_EQ(t.shape().size(), 2);
        t.ref_as<float, 2>();
        t.view_as<float, 2>();
    }
    {
        raw_tensor t(ttl::internal::typeinfo<float>(), 1, 2, 3);
        ASSERT_EQ(t.shape().size(), 6);
        t.ref_as<float, 3>();
        t.view_as<float, 3>();
    }
}
