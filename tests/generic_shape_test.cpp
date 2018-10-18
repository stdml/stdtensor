#include <gtest/gtest.h>

#include <bits/std_generic_shape.hpp>

TEST(generic_shape_test, test1)
{
    using generic_shape = ttl::internal::basic_generic_shape<std::uint32_t>;
    {
        generic_shape shape;
        ASSERT_EQ(shape.rank(), 0);
        ASSERT_EQ(shape.size(), 1);
        shape.as_ranked<0>();
    }
    {
        generic_shape shape(1);
        ASSERT_EQ(shape.rank(), 1);
        ASSERT_EQ(shape.size(), 1);
        shape.as_ranked<1>();
    }
    {
        generic_shape shape(1, 2);
        ASSERT_EQ(shape.rank(), 2);
        ASSERT_EQ(shape.size(), 2);
        shape.as_ranked<2>();
    }
    {
        generic_shape shape(1, 2, 3);
        ASSERT_EQ(shape.rank(), 3);
        ASSERT_EQ(shape.size(), 6);
        shape.as_ranked<3>();
    }
}
