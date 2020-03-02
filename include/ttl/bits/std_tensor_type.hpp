#pragma once
#include <cstddef>
#include <ttl/bits/std_except.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename S>
class basic_tensor_type : public S
{
    using S::S;

  public:
    const S &shape() const { return *this; }

    std::size_t data_size() const { return sizeof(R) * this->size(); }
};

template <typename E, typename S>
class basic_raw_tensor_type : public S
{
    using value_type_t = typename E::value_type;
    const value_type_t value_type_;

  public:
    template <typename R>
    static constexpr value_type_t type()
    {
        return E::template value<R>();
    }

    template <typename R>
    static basic_raw_tensor_type scalar()
    {
        return basic_raw_tensor_type(type<R>());
    }

    template <typename R, typename... Args>
    static basic_raw_tensor_type type(const Args &... args)
    {
        return basic_raw_tensor_type(type<R>(), args...);
    }

    template <typename... Args>
    explicit basic_raw_tensor_type(const value_type_t value_type,
                                   const Args &... args)
        : S(args...), value_type_(value_type)
    {
    }

    value_type_t value_type() const { return value_type_; }

    const S &shape() const { return *this; }

    std::size_t data_size() const
    {
        return E::size(value_type_) * this->size();
    }

    template <typename R>
    R *typed(void *data) const
    {
        if (type<R>() != value_type_) {
            throw invalid_type_reification(typeid(R));
        }
        return reinterpret_cast<R *>(data);
    }

    template <typename R>
    const R *typed(const void *data) const
    {
        if (type<R>() != value_type_) {
            throw invalid_type_reification(typeid(R));
        }
        return reinterpret_cast<const R *>(data);
    }
};
}  // namespace internal
}  // namespace ttl
