#include "testing.hpp"

#include <ttl/type_traits>

TEST(type_traits_test, test_1)
{
    using Ts = std::tuple<int, float>;
    {
        constexpr int pos = ttl::lookup<int, Ts>::value;
        ASSERT_EQ(pos, 0);
    }
    {
        constexpr int pos = ttl::lookup<float, Ts>::value;
        ASSERT_EQ(pos, 1);
    }
    {  // won't compile
       // constexpr int pos = ttl::lookup<double, Ts>::value;
       // ASSERT_EQ(pos, 2);
    }
}
