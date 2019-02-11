#pragma once

#define FAKE_CUDA_RUNTIME

template <typename T> void cudaMalloc(T **ptr, int count)
{
    *ptr = (T *)malloc(sizeof(T) * count);
}

void cudaFree(void *ptr) { free(ptr); }

constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;

void cudaMemcpy(void *dst, const void *src, int size, int direction)
{
    printf("%s::%s\n", __FILE__, __func__);
}
