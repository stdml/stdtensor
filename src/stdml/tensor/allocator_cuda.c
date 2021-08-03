#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*alloc_fn)(void **devPtr, size_t size);
typedef int (*free_fn)(void *addr);

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY

static const char *libcudart = "/usr/local/cuda/lib64/libcudart.so";

static alloc_fn get_alloc_fn()
{
    int mode = RTLD_LAZY;
    void *h = dlopen(libcudart, mode);
    if (h == 0) {
        printf("dlopen failed: %s\n", dlerror());  //
        exit(1);
    }
    alloc_fn f = dlsym(h, "cudaMalloc");
    return f;
}

static free_fn get_free_fn()
{
    int mode = RTLD_LAZY;
    void *h = dlopen(libcudart, mode);
    if (h == 0) {
        printf("dlopen failed: %s\n", dlerror());  //
        exit(1);
    }
    free_fn f = dlsym(h, "cudaFree");
    return f;
}

void *cuda_alloc(size_t size)
{
    alloc_fn f = get_alloc_fn();
    void *ptr = 0;
    int code = f(&ptr, size);
    if (code != 0) {
        perror("alloc failed");
        exit(1);
    }
    return ptr;
}

void cuda_free(void *ptr)
{
    free_fn f = get_free_fn();
    int code = f(ptr);
    if (code != 0) {
        perror("free failed");
        exit(1);
    }
}
