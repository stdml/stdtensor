#include "testing.hpp"
#include <ttl/experimental/type>
#include <ttl/flat_shape>
#include <ttl/shape>
#include <ttl/type>

TEST(type_test, test_type)
{
    ttl::type<double, 2> tt(3, 4);
    ASSERT_EQ(tt.size(), static_cast<uint32_t>(12));
    ASSERT_EQ(tt.shape(), ttl::make_shape(3, 4));
    ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(12 * 8));
}

namespace ttl
{
using experimental::raw_type;
}  // namespace ttl

TEST(type_test, test_raw_type)
{
    using TT = ttl::raw_type<>;
    {
        TT tt(TT::type<double>(), 3, 4);
        ASSERT_EQ(tt.size(), static_cast<uint32_t>(12));
        ASSERT_EQ(tt.shape(), ttl::flat_shape(3, 4));
        ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(12 * 8));
    }
    {
        TT tt = TT::scalar<double>();
        ASSERT_EQ(tt.size(), static_cast<uint32_t>(1));
        ASSERT_EQ(tt.shape(), ttl::flat_shape());
        ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(8));
    }
    {
        TT tt = TT::type<double>(3, 4);
        ASSERT_EQ(tt.size(), static_cast<uint32_t>(12));
        ASSERT_EQ(tt.shape(), ttl::flat_shape(3, 4));
        ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(12 * 8));
    }
}
