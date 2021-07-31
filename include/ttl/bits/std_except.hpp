#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

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
    // TODO: move this to library?
    template <typename Dim>
    static std::string msg(const std::vector<Dim> &dims, int rank)
    {
        std::stringstream ss;
        ss << '(';
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) { ss << ", "; }
            ss << dims[i];
        }
        ss << ')';
        ss << " as rank " << rank;
        return ss.str();
    }

  public:
    // invalid_rank_reification() : invalid_argument(__func__) {}

    template <typename Dim>
    invalid_rank_reification(const std::vector<Dim> &dims, int rank)
        : invalid_argument(msg(dims, rank))
    {
    }
};

class invalid_device_reification : public std::invalid_argument
{
  public:
    invalid_device_reification() : invalid_argument(__func__) {}
};
}  // namespace internal
}  // namespace ttl
