#include <string>

#include "testing.hpp"
#include <ttl/tensor>

template <typename F>
void for_all_types(const F &f)
{
    f.template operator()<char>();

    f.template operator()<uint8_t>();
    f.template operator()<uint16_t>();
    f.template operator()<uint32_t>();
    f.template operator()<uint64_t>();

    f.template operator()<int8_t>();
    f.template operator()<int16_t>();
    f.template operator()<int32_t>();
    f.template operator()<int64_t>();

    f.template operator()<float>();
    f.template operator()<double>();

    f.template operator()<std::string>();
}

template <typename F>
void for_idx_types(const F &f)
{
    f.template operator()<uint8_t>();

    f.template operator()<int8_t>();
    f.template operator()<int16_t>();
    f.template operator()<int32_t>();

    f.template operator()<float>();
    f.template operator()<double>();
}

template <ttl::rank_t r>
ttl::shape<r> unit_shape()
{
    std::array<typename ttl::shape<r>::dimension_type, r> dims;
    std::fill(dims.begin(), dims.end(), 1);
    return ttl::shape<r>(dims);
}

template <typename T>
void test_public_apis(const T &t)
{
    const auto size = t.size();
    ASSERT_EQ(size, static_cast<decltype(size)>(1));

    const auto dims = t.dims();
    static_assert(dims.size() == T::rank, "");
}

template <ttl::rank_t r>
struct test_ranked_type {
    template <typename R>
    void operator()() const
    {
        using Tensor = ttl::tensor<R, r>;
        using TensorRef = ttl::tensor_ref<R, r>;
        using TensorView = ttl::tensor_view<R, r>;

        // FIXME: assert size
        // using Dim = typename ttl::shape<r>::dimension_type;
        // constexpr size_t size = sizeof(void *) + r * sizeof(Dim);

        // static_assert(sizeof(Tensor) == size, "");
        // static_assert(sizeof(TensorRef) == size, "");
        // static_assert(sizeof(TensorView) == size, "");

        Tensor t(unit_shape<r>());
        TensorRef tr(t);
        TensorView tv(t);

        test_public_apis(t);
        test_public_apis(tr);
        test_public_apis(tv);
    }
};

TEST(public_types_test, test_ranks)
{
    for_all_types(test_ranked_type<0>());
    for_all_types(test_ranked_type<1>());
    for_all_types(test_ranked_type<2>());
    for_all_types(test_ranked_type<3>());
    for_all_types(test_ranked_type<4>());
    for_all_types(test_ranked_type<5>());
    for_all_types(test_ranked_type<6>());
    for_all_types(test_ranked_type<7>());
    for_all_types(test_ranked_type<8>());
    for_all_types(test_ranked_type<9>());
    for_all_types(test_ranked_type<10>());
}

#include <ttl/experimental/flat_tensor>

template <ttl::rank_t r>
struct test_flat_type {
    template <typename R>
    void operator()() const
    {
        using Tensor = ttl::experimental::flat_tensor<R>;
        using TensorRef = ttl::experimental::flat_tensor_ref<R>;
        using TensorView = ttl::experimental::flat_tensor_view<R>;

        // FIXME: assert size

        Tensor t(unit_shape<r>());
        TensorRef tr(t);
        TensorView tv(t);
    }
};

TEST(public_types_test, test_flat_types)
{
    for_all_types(test_flat_type<0>());
    for_all_types(test_flat_type<1>());
    for_all_types(test_flat_type<2>());
    for_all_types(test_flat_type<3>());
    for_all_types(test_flat_type<4>());
    for_all_types(test_flat_type<5>());
    for_all_types(test_flat_type<6>());
    for_all_types(test_flat_type<7>());
    for_all_types(test_flat_type<8>());
    for_all_types(test_flat_type<9>());
    for_all_types(test_flat_type<10>());
}

#include <ttl/experimental/raw_tensor>

template <ttl::rank_t r>
struct test_raw_type {
    template <typename R>
    void operator()() const
    {
        using Tensor = ttl::experimental::raw_tensor;
        using TensorRef = ttl::experimental::raw_tensor_ref;
        using TensorView = ttl::experimental::raw_tensor_view;

        using Encoder = Tensor::encoder_type;

        // FIXME: assert size

        Tensor t(Encoder::value<R>(), unit_shape<r>());
        TensorRef tr(t);
        TensorView tv(t);
    }
};

TEST(public_types_test, test_raw_types)
{
    // FIXME: for_all_types
    for_idx_types(test_raw_type<0>());
    for_idx_types(test_raw_type<1>());
    for_idx_types(test_raw_type<2>());
    for_idx_types(test_raw_type<3>());
    for_idx_types(test_raw_type<4>());
    for_idx_types(test_raw_type<5>());
    for_idx_types(test_raw_type<6>());
    for_idx_types(test_raw_type<7>());
    for_idx_types(test_raw_type<8>());
    for_idx_types(test_raw_type<9>());
    for_idx_types(test_raw_type<10>());
}
