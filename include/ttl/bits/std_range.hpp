#pragma once

namespace ttl
{
namespace internal
{
template <typename N> class range_t
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
    explicit range_t(N n) : from_(0), to_(n) {}

    explicit range_t(N m, N n) : from_(m), to_(n) {}

    iterator begin() const { return iterator(from_); }

    iterator end() const { return iterator(to_); }
};

template <typename N> range_t<N> range(N n) { return range_t<N>(n); }

template <typename N> range_t<N> range(N m, N n) { return range_t<N>(m, n); }

// TODO:
/*
template <ttl::rank_t r, typename T> auto range(const T &t)
{
    return range(std::get<r>(t.shape().dims()));
}
*/

}  // namespace internal
}  // namespace ttl
