#include "benchmark.hpp"

#ifdef USE_FAKE_CUDA_RUNTIME
#include "fake_cuda_runtime.h"
#endif

#include <ttl/cuda_tensor>

template <typename R, int n> struct bench_cuda_tensor {
    static void run(benchmark::State &state)
    {
        ttl::cuda_tensor<R, 1> m1(n);
        ttl::tensor<R, 1> m2(n);

        for (auto _ : state) {
            m1.fromHost(m2.data());
            m1.toHost(m2.data());
        }
    }
};

constexpr size_t Mi = 1 << 20;

static void bench_cuda_tensor_1Mi(benchmark::State &state)
{
    bench_cuda_tensor<char, Mi>::run(state);
}
BENCHMARK(bench_cuda_tensor_1Mi)->Unit(benchmark::kMicrosecond);

static void bench_cuda_tensor_10Mi(benchmark::State &state)
{
    bench_cuda_tensor<char, 10 * Mi>::run(state);
}
BENCHMARK(bench_cuda_tensor_10Mi)->Unit(benchmark::kMillisecond);

static void bench_cuda_tensor_100Mi(benchmark::State &state)
{
    bench_cuda_tensor<char, 100 * Mi>::run(state);
}
BENCHMARK(bench_cuda_tensor_100Mi)->Unit(benchmark::kMillisecond);
