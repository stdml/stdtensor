#include "testing.hpp"

#include <ttl/bits/std_tensor_buffer.hpp>
#include <ttl/debug>

namespace ttl
{
template <typename R, typename D = ttl::host_memory>
using tensor_buffer = internal::basic_tensor_buffer<R, D>;
}

TEST(tensor_buffer_test, test1)
{
    using S = typename ttl::tensor_buffer<int>::shape_type;
    std::vector<S> shapes;
    shapes.emplace_back(ttl::make_shape(28, 28, 10));
    shapes.push_back(S(ttl::make_shape(10)));
    ttl::tensor_buffer<int> tb(shapes);
    ASSERT_EQ(static_cast<int>(tb.size()), 2);
    tb[0].ranked<3>();
    tb[1].ranked<1>();
}
