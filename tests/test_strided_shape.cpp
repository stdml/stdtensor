#include "testing.hpp"

#include <ttl/tensor>

using namespace ttl::internal;

using dim_t = uint32_t;
template <rank_t r> using shape = basic_shape<r, dim_t>;
template <rank_t r> using sshape = basic_strided_shape<r, dim_t>;

TEST(strided_shape_test, test1)
{
    shape<3> s(4, 5, 6);
    sshape<3> ss(s, 1, 2, 3);
}
