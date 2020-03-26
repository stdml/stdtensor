#include "benchmark.hpp"

#include <ttl/cuda_tensor>
#include <ttl/experimental/copy>

template <typename R, size_t n>
struct bench_cuda_tensor {
    static void run(benchmark::State &state)
    {
        ttl::cuda_tensor<R, 1> m1(n);
        ttl::tensor<R, 1> m2(n);

        for (auto _ : state) {
            ttl::copy(ttl::ref(m1), ttl::view(m2));
            ttl::copy(ttl::ref(m2), ttl::view(m1));
        }
    }
};

constexpr size_t Mi = 1 << 20;
constexpr size_t Gi = 1 << 30;

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

static void bench_cuda_tensor_1Gi(benchmark::State &state)
{
    bench_cuda_tensor<char, 1 * Gi>::run(state);
}
BENCHMARK(bench_cuda_tensor_1Gi)->Unit(benchmark::kMillisecond);

static void bench_cuda_tensor_2Gi(benchmark::State &state)
{
    bench_cuda_tensor<char, 2 * Gi>::run(state);
}
BENCHMARK(bench_cuda_tensor_2Gi)->Unit(benchmark::kMillisecond);

static void bench_cuda_tensor_3Gi(benchmark::State &state)
{
    bench_cuda_tensor<char, 3 * Gi>::run(state);
}
BENCHMARK(bench_cuda_tensor_3Gi)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
