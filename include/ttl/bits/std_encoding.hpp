#pragma once
#include <cstdint>
#include <tuple>

namespace ttl
{
namespace internal
{
template <bool is_signed, bool is_float, typename V = uint32_t>
struct basic_scalar_category;

template <typename V>
struct basic_scalar_category<false, false, V> {
    static constexpr V value = 0;
};

template <typename V>
struct basic_scalar_category<true, false, V> {
    static constexpr V value = 1;
};

template <typename V>
struct basic_scalar_category<true, true, V> {
    static constexpr V value = 3;
};

template <typename R, typename V>
class basic_scalar_encoding
{
    static constexpr V category =
        basic_scalar_category<std::is_signed<R>::value,
                              std::is_floating_point<R>::value, V>::value;
    static constexpr V byte_num = sizeof(R);
    static constexpr V byte_size = CHAR_BIT;

  public:
    static constexpr V value = (category << 16) | (byte_num << 8) | byte_size;
};
}  // namespace internal

namespace experimental
{
struct std_encoding {
    using value_type = uint32_t;

    using types = std::tuple<                                       //
        std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t,  //
        std::int8_t, std::int16_t, std::int32_t, std::int64_t,      //
        float, double>;

    template <typename R>
    static constexpr value_type value()
    {
        return internal::basic_scalar_encoding<R, value_type>::value;
    }
};
}  // namespace experimental
}  // namespace ttl
