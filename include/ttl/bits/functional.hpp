#pragma once

namespace ttl
{
namespace internal
{
template <typename N = size_t>
struct length {
    template <typename T>
    N operator()(const T &t) const
    {
        return t.len();
    }
};

template <typename N = size_t>
class slice
{
    N a;
    N b;

  public:
    slice(N a, N b) : a(a), b(b) {}

    template <typename T>
    auto operator()(const T &t) const
    {
        return t.slice(a, b);
    }
};
}  // namespace internal
}  // namespace ttl
