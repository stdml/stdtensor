#pragma once
#include <stdexcept>
#include <typeinfo>

namespace ttl
{
namespace internal
{
class invalid_type_reification : public std::invalid_argument
{
  public:
    invalid_type_reification(const std::type_info &ti)
        : invalid_argument(ti.name())  // FIXME: demangling
    {
    }
};

class invalid_rank_reification : public std::invalid_argument
{
  public:
    invalid_rank_reification() : invalid_argument(__func__) {}
};

class invalid_device_reification : public std::invalid_argument
{
  public:
    invalid_device_reification() : invalid_argument(__func__) {}
};
}  // namespace internal
}  // namespace ttl
