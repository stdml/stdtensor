#include <cstdio>
#include <stdexcept>
#include <stdml/bits/dll.hpp>

#include <dlfcn.h>

namespace stdml
{
static constexpr const char *EXT =
#if defined(__APPLE__) && defined(__MACH__)
    ".dylib"
#else
    ".so"
#endif
    ;

static constexpr int mode = RTLD_LAZY;

dll::dll(std::string name, std::string prefix)
{
    std::string soname = "lib" + name + EXT;
    std::string path = prefix + soname;
    handle_ = dlopen(path.c_str(), mode);
    if (handle_ == nullptr) { throw std::runtime_error(dlerror()); }
}

void *dll::raw_sym(const std::string &name) const
{
    void *f = dlsym(handle_, name.c_str());
    if (f == nullptr) {
        fprintf(stderr, "%s not defined\n", name.c_str());
        throw std::runtime_error(std::string("dlsym failed: ") + dlerror());
    }
    return f;
}

bool try_dl_open(std::string path)
{
    void *handle = dlopen(path.c_str(), mode);
    if (handle == nullptr) { return false; }
    dlclose(handle);
    return true;
}
}  // namespace stdml
