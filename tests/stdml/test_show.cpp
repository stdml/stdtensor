#include <iostream>

#include "../testing.hpp"

#include <stdml/tensor>  // FIXME: use ttl/tensor

TEST(show_test, test_show)
{
    ttl::tensor<int, 2> x(2, 2);
    std::iota(x.data(), x.data_end(), 1);
    std::cout << ttl::show(ttl::view(x)) << std::endl;
}
