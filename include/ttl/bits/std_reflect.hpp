#pragma once
#include <cxxabi.h>
#include <string>

namespace ttl
{
namespace internal
{
template <typename T>
std::string demangled_type_info_name()
{
    int status = 0;
    return abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
}

template <typename R>
constexpr char scalar_type_prefix()
{
    if (std::is_floating_point<R>::value) {
        return 'f';
    } else if (std::is_integral<R>::value) {
        return std::is_signed<R>::value ? 'i' : 'u';
    } else {
        return 's';
    }
}

template <bool, typename R>
class scalar_type_name;

template <typename R>
class scalar_type_name<false, R>
{
  public:
    std::string operator()() const { return demangled_type_info_name<R>(); }
};

template <typename R>
class scalar_type_name<true, R>
{
  public:
    std::string operator()() const
    {
        return scalar_type_prefix<R>() + std::to_string(sizeof(R) * CHAR_BIT);
    }
};
}  // namespace internal
}  // namespace ttl
