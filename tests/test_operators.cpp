#include "testing.hpp"

#include <ttl/operators>
#include <ttl/tensor>

TEST(operators_test, test_1)
{
    ttl::tensor<int, 1> x(1);
    using namespace ttl::operators;
    auto r = &x;
    auto v1 = !x;
    auto v2 = !r;
    auto v3 = !&x;

    static_assert(std::is_same<decltype(r), ttl::tensor_ref<int, 1>>::value,
                  "");
    static_assert(std::is_same<decltype(v1), ttl::tensor_view<int, 1>>::value,
                  "");
    static_assert(std::is_same<decltype(v2), ttl::tensor_view<int, 1>>::value,
                  "");
    static_assert(std::is_same<decltype(v3), ttl::tensor_view<int, 1>>::value,
                  "");
}
