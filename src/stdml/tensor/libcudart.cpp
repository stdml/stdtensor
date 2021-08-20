#include <map>
#include <memory>
#include <stdexcept>

#include <stdml/bits/dll.hpp>
#include <stdml/bits/tensor/cudart.hpp>

namespace stdml
{
// static constexpr int Mi = 1 << 20;

class memstat
{
    int64_t allocated_;
    int64_t blocks_;

    int64_t peek_;
    int64_t peek_blocks_;

    std::map<const void *, int64_t> _allocs;

  public:
    memstat() : allocated_(0), blocks_(0), peek_(0), peek_blocks_(0) {}

    void in(int64_t n, const void *p)
    {
        allocated_ += n;
        ++blocks_;
        _allocs[p] = n;

        if (allocated_ > peek_) { peek_ = allocated_; }
        if (blocks_ > peek_blocks_) { peek_blocks_ = blocks_; }
        // fprintf(stderr,
        //         ">>>total allocated: %.2fMiB (peek: %.2fMiB), %d blocks
        //         (peek: "
        //         "%d)\n",
        //         (double)allocated_ / Mi, (double)peek_ / Mi, (int)blocks_,
        //         (int)peek_blocks_);
    }

    void out(const void *p)
    {
        if (_allocs.count(p) == 0) {
            fprintf(stderr, "invalid free\n");
            exit(0);
        } else {
            int64_t n = _allocs.at(p);
            allocated_ -= n;
            --blocks_;
            // fprintf(stderr, "...still allocated: %.2fMiB, %d blocks_\n",
            //         (double)allocated_ / Mi, (int)blocks_);
        }
    }
};

class libcudart_impl : public libcudart
{
    typedef int (*alloc_fn)(void **devPtr, size_t size);
    typedef int (*free_fn)(void *addr);
    typedef int (*copy_fn)(void *dst, const void *src, size_t size,
                           int /* cudaMemcpyKind */ dir);
    typedef int (*set_dev_fn)(int);
    typedef int (*get_dev_cnt_fn)(int *);
    typedef const char *(*get_err_str_fn)(int err);

    dll dll_;
    alloc_fn alloc_;
    free_fn free_;
    copy_fn copy_;
    set_dev_fn set_dev_;
    get_dev_cnt_fn get_dev_cnt_;
    get_err_str_fn get_err_str_;

    std::unique_ptr<memstat> ms_;

  public:
    libcudart_impl()
        : dll_("cudart", "/usr/local/cuda/lib64/"),
          alloc_(dll_.sym<alloc_fn>("cudaMalloc")),
          free_(dll_.sym<free_fn>("cudaFree")),
          copy_(dll_.sym<copy_fn>("cudaMemcpy")),
          set_dev_(dll_.sym<set_dev_fn>("cudaSetDevice")),
          get_dev_cnt_(dll_.sym<get_dev_cnt_fn>("cudaGetDeviceCount")),
          get_err_str_(dll_.sym<get_err_str_fn>("cudaGetErrorString")),
          ms_(new memstat)
    {
    }

    const char *get_err_str(int err) const override
    {
        return get_err_str_(err);
    }

    void *cuda_alloc(size_t size) const override
    {
        void *ptr = nullptr;
        if (int err = alloc_(&ptr, size); err != 0) {
            throw std::runtime_error("cudaMalloc() failed: " +
                                     std::string(get_err_str_(err)));
        }
        ms_->in(size, ptr);
        return ptr;
    }

    void cuda_free(void *p) const override
    {
        if (int err = free_(p); err != 0) {
            throw std::runtime_error("cudaFree() failed: " +
                                     std::string(get_err_str_(err)));
        }
        ms_->out(p);
    }

    // cudaMemcpy won't return error when dir = 0
    void _copy(void *dst, const void *src, size_t n, int dir) const
    {
        if (int err = copy_(dst, src, n, dir); err != 0) {
            throw std::runtime_error("cudaMemcpy() failed: " +
                                     std::string(get_err_str_(err)));
        }
    }

    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
    static constexpr int cudaMemcpyHostToDevice = 1;
    static constexpr int cudaMemcpyDeviceToHost = 2;
    static constexpr int cudaMemcpyDeviceToDevice = 3;

    void from_host(void *dst, const void *src, size_t n) const override
    {
        _copy(dst, src, n, cudaMemcpyHostToDevice);
    }

    void to_host(void *dst, const void *src, size_t n) const override
    {
        _copy(dst, src, n, cudaMemcpyDeviceToHost);
    }

    void d2d(void *dst, const void *src, size_t n) const override
    {
        _copy(dst, src, n, cudaMemcpyDeviceToDevice);
    }

    void set_cuda_device(int d) const override
    {
        if (int err = set_dev_(d); err != 0) {
            throw std::runtime_error("cudaSetDevice() failed: " +
                                     std::string(get_err_str_(err)));
        }
    }

    int get_gpu_count() const override
    {
        int cnt = 0;
        if (int err = get_dev_cnt_(&cnt); err != 0) {
            throw std::runtime_error("cudaGetDeviceCount() failed: " +
                                     std::string(get_err_str_(err)));
        }
        return cnt;
    }
};

libcudart &libcudart::get()
{
    static libcudart_impl instance;
    return instance;
}

bool has_cuda()
{
    static bool ok = try_dl_open("/usr/local/cuda/lib64/libcudart.so");
    return ok;
}

int get_cuda_gpu_count()
{
    if (!has_cuda()) { return 0; }
    return libcudart::get().get_gpu_count();
}
}  // namespace stdml
