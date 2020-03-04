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
}  // namespace internal
}  // namespace ttl
