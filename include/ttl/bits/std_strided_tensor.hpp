#pragma once
#include <memory>

#include <ttl/bits/std_strided_shape.hpp>

namespace ttl
{
namespace internal
{

template <typename R, rank_t r, typename shape_t>
class basic_strided_tensor_ref;

template <typename R, rank_t r, typename shape_t = basic_strided_shape<r>>
class basic_strided_tensor_ref
{
  public:
    using value_type = R;

    static constexpr rank_t rank = r;
};

}  // namespace internal
}  // namespace ttl
