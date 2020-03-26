#pragma once

#if defined(__GNUC__) && !defined(__clang__)
#pragma message("ttl::experimental::zip is error-prone, use with care!")
#endif

#include <array>
#include <functional>
#include <numeric>
#include <tuple>

namespace ttl
{
namespace experimental
{
namespace internal
{
template <typename... Ts>
class zipper_t
{
    static constexpr auto arity = sizeof...(Ts);

    const std::tuple<const Ts &...> ranges_;

    template <typename... Iters>
    class iterator
    {
        std::tuple<Iters...> is_;

        template <size_t... Is>
        auto operator*(std::index_sequence<Is...>)
        {
            return std::make_tuple(*std::get<Is>(is_)...);
        }

        template <typename... P>
        static void noop(const P &...)
        {
        }

        template <typename Iter>
        int incr(Iter &i)
        {
            ++i;
            return 0;
        }

        template <size_t... Is>
        void _advance(std::index_sequence<Is...>)
        {
            noop(incr(std::get<Is>(is_))...);
        }

        template <size_t... Is>
        bool neq(std::index_sequence<Is...>, const iterator &p) const
        {
            // TODO: expand the expression
            std::array<bool, arity> neqs(
                {(std::get<Is>(is_) != std::get<Is>(p.is_))...});
            return std::accumulate(neqs.begin(), neqs.end(), false,
                                   std::logical_or<bool>());
        }

      public:
        iterator(const Iters &... i) : is_(i...) {}

        bool operator!=(const iterator &p) const
        {
            // return get<0>(is_) != get<0>(p.is_) || get<1>(is_) !=
            // get<1>(p.is_);
            return neq(std::make_index_sequence<arity>(), p);
        }

        void operator++()
        {
            _advance(std::make_index_sequence<arity>());
            // ++get<0>(is_), ++get<1>(is_);
        }

        auto operator*()
        {
            return (operator*)(std::make_index_sequence<arity>());
        }
    };

    template <typename... Iters>
    static iterator<Iters...> make_iterator(const Iters &... is)
    {
        return iterator<Iters...>(is...);
    }

    template <size_t... Is>
    auto begin(std::index_sequence<Is...>) const
    {
        return make_iterator(std::get<Is>(ranges_).begin()...);
    }

    template <size_t... Is>
    auto end(std::index_sequence<Is...>) const
    {
        return make_iterator(std::get<Is>(ranges_).end()...);
    }

  public:
    zipper_t(const Ts &... ranges) : ranges_(ranges...) {}

    auto begin() const { return begin(std::make_index_sequence<arity>()); }

    auto end() const { return end(std::make_index_sequence<arity>()); }
};

template <typename... Ts>
zipper_t<Ts...> zip(const Ts &... ranges)
{
    return zipper_t<Ts...>(ranges...);
}
}  // namespace internal
}  // namespace experimental
}  // namespace ttl
