#include "testing.hpp"
#include <ttl/tensor>

template <ttl::rank_t r>
ttl::shape<r> unit_shape()
{
    std::array<typename ttl::shape<r>::dimension_type, r> dims;
    std::fill(dims.begin(), dims.end(), 1);
    return ttl::shape<r>(dims);
}

template <typename R, ttl::rank_t r>
void test_type()
{
    using Tensor = ttl::tensor<R, r>;
    using TensorRef = ttl::tensor_ref<R, r>;
    using TensorView = ttl::tensor_view<R, r>;

    // using Dim = typename ttl::shape<r>::dimension_type;
    // constexpr size_t size = sizeof(void *) + r * sizeof(Dim);

    // static_assert(sizeof(Tensor) == size, "");
    // static_assert(sizeof(TensorRef) == size, "");
    // static_assert(sizeof(TensorView) == size, "");

    Tensor t(unit_shape<r>());
    TensorRef tr(t);
    TensorView tv(t);
}

template <ttl::rank_t r>
void test_rank()
{
    test_type<char, r>();

    test_type<uint8_t, r>();
    test_type<uint16_t, r>();
    test_type<uint32_t, r>();
    test_type<uint64_t, r>();

    test_type<int8_t, r>();
    test_type<int16_t, r>();
    test_type<int32_t, r>();
    test_type<int64_t, r>();

    test_type<float, r>();
    test_type<double, r>();
}

TEST(public_types_test, test_types)
{
    test_rank<0>();
    test_rank<1>();
    test_rank<2>();
    test_rank<3>();
    test_rank<4>();
    test_rank<5>();
    test_rank<6>();
    test_rank<7>();
    test_rank<8>();
    test_rank<9>();
    test_rank<10>();
}
