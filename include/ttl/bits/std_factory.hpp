#pragma once
#include <functional>
#include <map>

namespace ttl
{
namespace internal
{
template <typename T, typename A>
using basic_constructor_t = std::function<T(A)>;

template <typename T, typename C, typename A>
struct basic_new_ptr_constructor {
    T *operator()(A a) const { return new C(std::move(a)); }
};

template <typename T, typename A, typename K>
class basic_factory_t
{
    using constructor = constructor_t<T, A>;
    std::map<K, constructor> constructors_;

  public:
    void operator()(K k, constructor c)
    {
        if (constructors_.count(k) > 0) {
            throw std::invalid_argument("registering duplicated constructor");
        }
        constructors_.emplace(std::move(k), std::move(c));
    }

    bool contains(K k) const { return constructors_.count(k) > 0; }

    constructor operator[](K k) const { return constructors_.at(k); }

    T operator()(K k, A a) const
    {
        if (constructors_.count(k) <= 0) {
            throw std::invalid_argument(
                "constructor not registered");  // FIXME: show k
        }
        return constructors_.at(k)(std::move(a));
    }
};
}  // namespace internal
}  // namespace ttl
