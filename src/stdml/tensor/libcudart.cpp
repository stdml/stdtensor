#include <stdml/bits/dll.hpp>
#include <stdml/bits/tensor/cudart.hpp>

namespace stdml
{
class libcudart_impl : public libcudart
{
    typedef int (*alloc_fn)(void **devPtr, size_t size);
    typedef int (*free_fn)(void *addr);

    typedef int (*copy_fn)(void *dst, const void *src, size_t size,
                           int /* cudaMemcpyKind */ dir);

    typedef const char *(*get_err_str_fn)(int err);

    dll dll_;
    alloc_fn alloc_fn_;
    free_fn free_fn_;
    copy_fn copy_fn_;
    get_err_str_fn get_err_str_;

  public:
    libcudart_impl()
        : dll_("cudart", "/usr/local/cuda/lib64/"),
          alloc_fn_(dll_.sym<alloc_fn>("cudaMalloc")),
          free_fn_(dll_.sym<free_fn>("cudaFree")),
          copy_fn_(dll_.sym<copy_fn>("cudaMemcpy")),
          get_err_str_(dll_.sym<get_err_str_fn>("cudaGetErrorString"))
    {
    }

    void *cuda_alloc(size_t size) const override
    {
        void *ptr = nullptr;
        if (int err = alloc_fn_(&ptr, size); err != 0) {
            throw std::runtime_error("cudaMalloc() failed: " +
                                     std::string(get_err_str_(err)));
        }
        return ptr;
    }

    void cuda_free(void *p) const override
    {
        if (int err = free_fn_(p); err != 0) {
            throw std::runtime_error("cudaFree() failed: " +
                                     std::string(get_err_str_(err)));
        }
    }

    void from_host(void *dst, const void *src, size_t n) const override
    {
        int dir = 0;
        if (int err = copy_fn_(dst, src, n, dir); err != 0) {
            throw std::runtime_error("cudaMemcpy() failed: " +
                                     std::string(get_err_str_(err)));
        }
    }

    void to_host(void *dst, const void *src, size_t n) const override
    {
        int dir = 0;
        if (int err = copy_fn_(dst, src, n, dir); err != 0) {
            throw std::runtime_error("cudaMemcpy() failed: " +
                                     std::string(get_err_str_(err)));
        }
    }
};

libcudart &libcudart::get()
{
    static libcudart_impl instance;
    return instance;
}
}  // namespace stdml
