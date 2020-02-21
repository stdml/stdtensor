#include "testing.hpp"

#include <ttl/bits/flat_shape.hpp>

TEST(flat_shape_test, test1)
{
    using flat_shape = ttl::internal::basic_flat_shape<std::uint32_t>;
    {
        flat_shape shape;
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(0));
        ASSERT_EQ(shape.size(), static_cast<flat_shape::dimension_type>(1));
        shape.ranked<0>();
    }
    {
        flat_shape shape(1);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(1));
        ASSERT_EQ(shape.size(), static_cast<flat_shape::dimension_type>(1));
        shape.ranked<1>();
    }
    {
        flat_shape shape(1, 2);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(2));
        ASSERT_EQ(shape.size(), static_cast<flat_shape::dimension_type>(2));
        shape.ranked<2>();
    }
    {
        flat_shape shape(1, 2, 3);
        ASSERT_EQ(shape.rank(), static_cast<ttl::internal::rank_t>(3));
        ASSERT_EQ(shape.size(), static_cast<flat_shape::dimension_type>(6));
        shape.ranked<3>();
    }
}

TEST(flat_shape_test, test_dims)
{
    using flat_shape = ttl::internal::basic_flat_shape<uint32_t>;
    flat_shape shape(2, 3, 4);
    auto dims = shape.dims();

    using dim_t = flat_shape::dimension_type;
    ASSERT_EQ(dims[0], static_cast<dim_t>(2));
    ASSERT_EQ(dims[1], static_cast<dim_t>(3));
    ASSERT_EQ(dims[2], static_cast<dim_t>(4));
}
