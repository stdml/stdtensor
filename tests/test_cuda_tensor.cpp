#include "testing.hpp"

#include <ttl/cuda_tensor>
#include <ttl/range>
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
    static_assert(sizeof(T0) == sizeof(void *), "");
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

template <typename R, uint8_t r> void test_auto_ref()
{
    static_assert(
        std::is_convertible<cuda_tensor<R, r>, cuda_tensor_ref<R, r>>::value,
        "can't convert to ref");
}

template <typename R, uint8_t r> void test_auto_view()
{
    static_assert(
        std::is_convertible<cuda_tensor<R, r>, cuda_tensor_view<R, r>>::value,
        "can't convert to view");

    static_assert(std::is_convertible<cuda_tensor_ref<R, r>,
                                      cuda_tensor_view<R, r>>::value,
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

template <typename R, uint8_t r> void test_copy(const ttl::shape<r> &shape)
{
    tensor<R, r> x(shape);
    cuda_tensor<R, r> y(shape);
    tensor<R, r> z(shape);

    std::iota(x.data(), x.data_end(), 1);
    y.from_host(x.data());
    y.to_host(z.data());

    for (auto i : ttl::range(shape.size())) {
        ASSERT_EQ(x.data()[i], z.data()[i]);
    }

    {
        cuda_tensor_ref<R, r> ry = ref(y);
        ry.from_host(x.data());
        ry.to_host(x.data());
    }
    {
        cuda_tensor_view<R, r> vy = view(y);
        vy.to_host(x.data());
    }
}

TEST(cuda_tensor_test, test_copy)
{
    test_copy<int, 0>(ttl::make_shape());
    test_copy<int, 1>(ttl::make_shape(10));
    test_copy<int, 2>(ttl::make_shape(4, 5));
    test_copy<int, 3>(ttl::make_shape(2, 3, 4));
}
