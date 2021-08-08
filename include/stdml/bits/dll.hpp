#pragma once
#include <string>

namespace stdml
{
class dll
{
    void *handle_;

  public:
    dll(std::string name, std::string prefix = "");

    void *raw_sym(const std::string &name) const;

    template <typename T>
    T sym(const std::string &name) const
    {
        return reinterpret_cast<T>(raw_sym(name));
    }
};

bool try_dl_open(std::string path);
}  // namespace stdml
