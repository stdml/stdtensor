#pragma once
#include <cstdint>

namespace ttl
{
namespace internal
{
using rank_t = uint8_t;

template <typename R, rank_t r, typename I, typename D>
class basic_sparse_tensor;
}  // namespace internal
}  // namespace ttl
