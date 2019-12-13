#include "testing.hpp"

#include <type_traits>

#include <ttl/tensor>

TEST(tensor_reshape_test, test_chunk)
{
    int n = 12;
    ttl::tensor<int, 2> x(n, 5);
    auto y = ttl::chunk(x, 3);
    auto z = ttl::flatten(ttl::view(y));

    static_assert(std::is_same<decltype(y), ttl::tensor_ref<int, 3>>::value,
                  "");
    static_assert(std::is_same<decltype(z), ttl::tensor_view<int, 1>>::value,
                  "");
}
