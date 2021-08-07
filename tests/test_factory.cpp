#include "testing.hpp"

#include <ttl/bits/std_factory.hpp>

TEST(factory_test, test_1)
{
    using F = ttl::internal::basic_factory_t<int, int, std::string>;
    F f;
    f("x", [](int x) { return x + 1; });

    int y = f("x", 1);
    ASSERT_EQ(y, 2);
    ASSERT_EQ(f["x"](2), 3);
}
