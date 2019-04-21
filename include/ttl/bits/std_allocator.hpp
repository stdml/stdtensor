#pragma once
#include <memory>

namespace ttl
{
namespace internal
{
template <typename R> using own_ptr = std::unique_ptr<R[]>;

template <typename R> class ref_ptr
{
    R *ptr_;

  public:
    ref_ptr(R *ptr) : ptr_(ptr) {}

    R *get() const { return ptr_; }
};

template <typename R> class view_ptr
{
    const R *ptr_;

  public:
    view_ptr(R *ptr) : ptr_(ptr) {}

    const R *get() const { return ptr_; }
};

}  // namespace internal
}  // namespace ttl
