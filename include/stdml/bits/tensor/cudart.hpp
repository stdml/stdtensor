#pragma once
#include <cstddef>

namespace stdml
{
class libcudart
{
  public:
    ~libcudart() = default;
    virtual void *cuda_alloc(size_t size) const = 0;
    virtual void cuda_free(void *p) const = 0;
    virtual void from_host(void *dst, const void *src, size_t n) const = 0;
    virtual void to_host(void *dst, const void *src, size_t n) const = 0;
    static libcudart &get();
};
}  // namespace stdml
