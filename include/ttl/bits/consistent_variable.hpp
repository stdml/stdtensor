#pragma once
#include <stdexcept>
#include <string>

namespace ttl
{
namespace internal
{
struct std_to_string {
    template <typename T>
    std::string operator()(const T &x) const
    {
        return std::to_string(x);
    }
};

template <typename T, typename Str = std_to_string>
class consistent_variable
{
    T value_;
    bool assigned_;

  public:
    consistent_variable() : assigned_(false) {}

    consistent_variable(T value) : value_(std::move(value)), assigned_(true) {}

    void operator=(T v)
    {
        if (assigned_) {
            if (value_ != v) {
                // TODO: customized exception
                throw std::invalid_argument(
                    std::string("consistent_variable was assigned: ") +
                    Str()(value_) + " now assigned: " + Str()(v));
            }
        } else {
            value_ = std::move(v);
            assigned_ = true;
        }
    }

    void operator=(T v) const
    {
        if (static_cast<T>(*this) != v) {
            throw std::invalid_argument(
                std::string("consistent_variable was assigned: ") +
                Str()(value_) + " now assigned: " + Str()(v));
        }
    }

    operator T() const
    {
        if (!assigned_) {
            // TODO: customized exception
            throw std::runtime_error("consistent_variable not assigned");
        }
        return value_;
    }
};
}  // namespace internal
}  // namespace ttl
