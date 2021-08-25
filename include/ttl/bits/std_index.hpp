#pragma once
#include <ttl/bits/std_tensor_fwd.hpp>
#include <ttl/device>
#include <ttl/shape>

namespace ttl
{
namespace internal
{
template <rank_t r, typename I = uint32_t>
class basic_index
{
    std::array<I, r> offsets_;

  public:
    static constexpr rank_t rank = r;

    template <typename... D>
    constexpr explicit basic_index(D... d) : offsets_({static_cast<I>(d)...})
    {
        static_assert(sizeof...(D) == r, "invalid number of indexes");
    }

    const std::array<I, r> &offsets() const { return offsets_; }
};

template <typename... I>
constexpr basic_index<sizeof...(I)> idx(const I... i)
{
    return basic_index<sizeof...(I)>(i...);
}
}  // namespace internal

using internal::idx;
}  // namespace ttl
