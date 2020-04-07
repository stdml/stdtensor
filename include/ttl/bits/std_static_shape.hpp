#pragma once
#include <utility>

namespace ttl
{
namespace internal
{
template <typename D, D... ds>
struct int_seq_prod;

template <typename D>
struct int_seq_prod<D> {
    static constexpr D value = 1;
};

template <typename D, D d0, D... ds>
struct int_seq_prod<D, d0, ds...> {
    static constexpr D value = d0 * int_seq_prod<D, ds...>::value;
};

template <typename D, D... ds>
class basic_static_shape : public std::integer_sequence<D, ds...>
{
  public:
    static constexpr auto rank = sizeof...(ds);
    static constexpr D dim = int_seq_prod<D, ds...>::value;
};
}  // namespace internal
}  // namespace ttl
