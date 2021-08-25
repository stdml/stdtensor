#pragma once
#include <tuple>

namespace ttl
{
namespace internal
{
template <bool p, typename T, T x, T y>
struct conditional;

template <typename T, T x, T y>
struct conditional<true, T, x, y> {
    static constexpr T value = x;
};

template <typename T, T x, T y>
struct conditional<false, T, x, y> {
    static constexpr T value = y;
};

template <typename T, typename Tuple>
struct rec_lookup;

template <typename T>
struct rec_lookup<T, std::tuple<>> {
    static constexpr size_t value = 0;
};

template <typename T, typename T0, typename... Ts>
struct rec_lookup<T, std::tuple<T0, Ts...>> {
    static constexpr size_t value =
        conditional<std::is_same<T, T0>::value, int, 0,
                    1 + rec_lookup<T, std::tuple<Ts...>>::value>::value;
};

template <typename T, typename Tuple>
struct lookup {
    static constexpr size_t value = rec_lookup<T, Tuple>::value;
    static_assert(value < std::tuple_size<Tuple>::value, "");
};
}  // namespace internal
}  // namespace ttl
