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
    ASSERT_EQ(tt.name(), "f64[3,4]");
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
        ASSERT_EQ(tt.name(), "f64[3,4]");
    }
    {
        TT tt = TT::scalar<double>();
        ASSERT_EQ(tt.size(), static_cast<uint32_t>(1));
        ASSERT_EQ(tt.shape(), ttl::flat_shape());
        ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(8));
        ASSERT_EQ(tt.name(), "f64[]");
    }
    {
        TT tt = TT::type<double>(3, 4);
        ASSERT_EQ(tt.size(), static_cast<uint32_t>(12));
        ASSERT_EQ(tt.shape(), ttl::flat_shape(3, 4));
        ASSERT_EQ(tt.data_size(), static_cast<uint32_t>(12 * 8));
    }
}

/*
TEST(type_test, test_type_str)
{
    ttl::type<std::string, 2> tt(3, 4);
    ASSERT_EQ(tt.name(),
              "std::__1::basic_string<char, std::__1::char_traits<char>, "
              "std::__1::allocator<char> >[3,4]");
    // using TT = ttl::raw_type<>;
    // TT tt(TT::type<std::string>(), 3, 4);
}
// */

TEST(type_test, test_type_reification)
{
    using TT = ttl::raw_type<>;
    const TT tt = TT::scalar<int>();
    {
        int x = 1;
        void *ptr = &x;
        int *iptr = tt.typed<int>(ptr);
        *iptr = 2;
        ASSERT_EQ(x, 2);
    }
    {
        const int x = 1;
        const void *ptr = &x;
        const int *iptr = tt.typed<int>(ptr);
        ASSERT_EQ(*iptr, 1);
    }
}
