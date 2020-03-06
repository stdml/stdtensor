#pragma once
#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <ttl/bits/std_reflect.hpp>

namespace ttl
{
namespace internal
{
template <typename Ts, typename P, typename E, std::size_t... I>
static constexpr std::array<P, sizeof...(I)>
    get_type_sizes(std::index_sequence<I...>)
{
    return {P{
        E::template value<typename std::tuple_element<I, Ts>::type>(),
        sizeof(typename std::tuple_element<I, Ts>::type),
    }...};
}

template <typename Ts, typename P, typename E, std::size_t... I>
static constexpr std::array<P, sizeof...(I)>
    get_type_prefixes(std::index_sequence<I...>)
{
    return {P{
        E::template value<typename std::tuple_element<I, Ts>::type>(),
        scalar_type_prefix<typename std::tuple_element<I, Ts>::type>(),
    }...};
}

template <typename encoding>
class basic_type_encoder
{
  public:
    using value_type = typename encoding::value_type;

    template <typename R>
    static constexpr value_type value()
    {
        return encoding::template value<R>();
    }

    static std::size_t size(const value_type type)
    {
        static constexpr int N =
            std::tuple_size<typename encoding::types>::value;
        using P = std::pair<value_type, std::size_t>;

        static constexpr std::array<P, N> type_sizes =
            get_type_sizes<typename encoding::types, P, encoding>(
                std::make_index_sequence<N>());

        for (int i = 0; i < N; ++i) {
            if (type_sizes[i].first == type) { return type_sizes[i].second; }
        }
        throw std::invalid_argument("invalid scalar code");
    }

    static char prefix(const value_type type)
    {
        static constexpr int N =
            std::tuple_size<typename encoding::types>::value;
        using P = std::pair<value_type, char>;

        static constexpr std::array<P, N> type_prefixes =
            get_type_prefixes<typename encoding::types, P, encoding>(
                std::make_index_sequence<N>());

        for (int i = 0; i < N; ++i) {
            if (type_prefixes[i].first == type) {
                return type_prefixes[i].second;
            }
        }
        throw std::invalid_argument("invalid scalar code");
    }
};
}  // namespace internal
}  // namespace ttl
