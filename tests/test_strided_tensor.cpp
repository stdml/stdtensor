#include "testing.hpp"

#include <ttl/tensor>

using ttl::experimental::sshape;

template <ttl::internal::rank_t r>
using shape = ttl::internal::basic_shape<r, typename sshape<r>::dimension_type>;

TEST(strided_tensor_test, test1)
{
    shape<3> s(4, 5, 6);
    sshape<3> ss(s, 1, 2, 3);
}
