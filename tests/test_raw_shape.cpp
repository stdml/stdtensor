#include "testing.hpp"

#include <ttl/bits/raw_shape.hpp>

TEST(raw_shape_test, test1)
{
    using raw_shape = ttl::internal::basic_raw_shape<std::uint32_t>;
    {
        raw_shape shape;
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(0));
        ASSERT_EQ(shape.size(), static_cast<raw_shape::dimension_type>(1));
        shape.as_ranked<0>();
    }
    {
        raw_shape shape(1);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(1));
        ASSERT_EQ(shape.size(), static_cast<raw_shape::dimension_type>(1));
        shape.as_ranked<1>();
    }
    {
        raw_shape shape(1, 2);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(2));
        ASSERT_EQ(shape.size(), static_cast<raw_shape::dimension_type>(2));
        shape.as_ranked<2>();
    }
    {
        raw_shape shape(1, 2, 3);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(3));
        ASSERT_EQ(shape.size(), static_cast<raw_shape::dimension_type>(6));
        shape.as_ranked<3>();
    }
}

TEST(raw_shape_test, test_dims)
{
    using raw_shape = ttl::internal::basic_raw_shape<uint32_t>;
    raw_shape shape(2, 3, 4);
    auto dims = shape.dims();

    using dim_t = raw_shape::dimension_type;
    ASSERT_EQ(dims[0], static_cast<dim_t>(2));
    ASSERT_EQ(dims[1], static_cast<dim_t>(3));
    ASSERT_EQ(dims[2], static_cast<dim_t>(4));
}
