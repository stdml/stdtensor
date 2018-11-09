#pragma once
#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>

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

template <typename encoding> class basic_type_encoder
{
  public:
    using data_type = typename encoding::data_type;

  private:
    using all_types = std::tuple<std::uint8_t, std::int8_t, std::int16_t,
                                 std::int32_t, float, double>;

    static constexpr int N = std::tuple_size<all_types>::value;

    using P = std::pair<data_type, std::size_t>;

  public:
    template <typename R> static constexpr data_type value()
    {
        return encoding::template value<R>();
    }

    static std::size_t size(const data_type type)
    {
        static constexpr std::array<P, N> type_sizes =
            get_type_sizes<all_types, P, encoding>(
                std::make_index_sequence<N>());

        for (int i = 0; i < N; ++i) {
            if (type_sizes[i].first == type) { return type_sizes[i].second; }
        }
        throw std::invalid_argument("invalid scalar code");
    }
};

}  // namespace internal
}  // namespace ttl
