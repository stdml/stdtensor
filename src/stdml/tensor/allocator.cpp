#include <cstdlib>
#include <stdexcept>
#include <stdml/bits/dll.hpp>
#include <stdml/bits/tensor/allocator.hpp>

namespace stdml
{
class libcudart
{
    typedef int (*alloc_fn)(void **devPtr, size_t size);
    typedef int (*free_fn)(void *addr);

    typedef int (*copy_fn)(void *dst, const void *src, size_t size,
                           int /* cudaMemcpyKind */ dir);

    dll dll_;
    alloc_fn alloc_fn_;
    free_fn free_fn_;
    copy_fn copy_fn_;

  public:
    libcudart()
        : dll_("cudart", "/usr/local/cuda/lib64/"),
          alloc_fn_(dll_.sym<alloc_fn>("cudaMalloc")),
          free_fn_(dll_.sym<free_fn>("cudaFree")),
          copy_fn_(dll_.sym<copy_fn>("cudaMemcpy"))
    {
    }

    void *cuda_alloc(size_t size) const
    {
        void *ptr = nullptr;
        if (int code = alloc_fn_(&ptr, size); code != 0) {
            throw std::runtime_error("cudaMalloc failed: ?");
        }
        return ptr;
    }

    void cuda_free(void *p) const { free_fn_(p); }

    static libcudart &get()
    {
        static libcudart instance;
        return instance;
    }
};

bool try_dl_open(std::string path);

bool has_cuda() { return try_dl_open("/usr/local/cuda/lib64/libcudart.so"); }

void *generic_allocator::alloc(Device device, size_t size)
{
    switch (device) {
    case cpu:
        return ::malloc(size);
    case cuda:
        return libcudart::get().cuda_alloc(size);
    default:
        throw std::runtime_error("invalid device!");
    }
}

void generic_allocator::free(Device device, void *addr)
{
    // printf("generic_allocator::free(%s, %p)\n", device_name(device), addr);
    switch (device) {
    case cpu:
        ::free(addr);
        return;
    case cuda:
        libcudart::get().cuda_free(addr);
        return;
    default:
        throw std::runtime_error("invalid device!");
    }
}

template <Device Dst, Device Src>
void generic_copier<Dst, Src>::operator()(void *dst, const void *src,
                                          size_t size) const
{
}
}  // namespace stdml
