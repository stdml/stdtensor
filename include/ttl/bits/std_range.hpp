#pragma once

namespace ttl
{
namespace internal
{
template <typename N>
class basic_integer_range
{
    const N from_;
    const N to_;

    class iterator
    {
        N pos_;

      public:
        explicit iterator(N pos) : pos_(pos) {}

        bool operator!=(const iterator &it) const { return pos_ != it.pos_; }

        N operator*() const { return pos_; }

        void operator++() { ++pos_; }
    };

  public:
    explicit basic_integer_range(N n) : from_(0), to_(n) {}

    explicit basic_integer_range(N m, N n) : from_(m), to_(n) {}

    iterator begin() const { return iterator(from_); }

    iterator end() const { return iterator(to_); }
};

template <typename N>
basic_integer_range<N> range(N n)
{
    return basic_integer_range<N>(n);
}

template <typename N>
basic_integer_range<N> range(N m, N n)
{
    return basic_integer_range<N>(m, n);
}
}  // namespace internal
}  // namespace ttl
