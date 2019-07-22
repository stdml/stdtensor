#include "testing.hpp"

#include <ttl/cuda_tensor>
#include <ttl/tensor>

using ttl::cuda_tensor;
using ttl::cuda_tensor_ref;
using ttl::cuda_tensor_view;
using ttl::tensor;

TEST(cuda_tensor_test, test_size)
{
    using T0 = ttl::cuda_tensor<uint8_t, 0>;
    using R0 = ttl::cuda_tensor_ref<uint8_t, 0>;
    using V0 = ttl::cuda_tensor_view<uint8_t, 0>;
    static_assert(sizeof(T0) == 2 * sizeof(void *), "");  // FIXME:
    static_assert(sizeof(R0) == sizeof(void *), "");
    static_assert(sizeof(V0) == sizeof(void *), "");
}

TEST(cuda_tensor_test, test0)
{
    using R = float;
    cuda_tensor<R, 0> m0;

    tensor<R, 0> x;

    m0.from_host(x.data());
    m0.to_host(x.data());
}

TEST(cuda_tensor_test, test1)
{
    using R = float;
    cuda_tensor<R, 2> m1(10, 100);
}

TEST(cuda_tensor_test, test2)
{
    using R = float;
    cuda_tensor<R, 2> m1(10, 100);
    tensor<R, 2> m2(10, 100);

    m1.from_host(m2.data());
    m1.to_host(m2.data());

    m1.slice(1, 2);
    auto r = ref(m1);
    UNUSED(r);
    auto v = view(m1);
    UNUSED(v);
}

TEST(cuda_tensor_test, test_3)
{
    using R = float;
    cuda_tensor<R, 2> m1(ttl::make_shape(10, 100));
}

template <typename T, uint8_t r> void test_auto_ref()
{
    static_assert(
        std::is_convertible<cuda_tensor<T, r>, cuda_tensor_ref<T, r>>::value,
        "can't convert to ref");
}

template <typename T, uint8_t r> void test_auto_view()
{
    static_assert(
        std::is_convertible<cuda_tensor<T, r>, cuda_tensor_view<T, r>>::value,
        "can't convert to view");

    static_assert(std::is_convertible<cuda_tensor_ref<T, r>,
                                      cuda_tensor_view<T, r>>::value,
                  "can't convert to view");
}

TEST(cuda_tensor_test, test_convert)
{
    test_auto_ref<int, 0>();
    test_auto_ref<int, 1>();
    test_auto_ref<int, 2>();

    test_auto_view<int, 0>();
    test_auto_view<int, 1>();
    test_auto_view<int, 2>();
}
