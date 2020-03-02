#include "testing.hpp"

#include <ttl/bits/std_encoding.hpp>
#include <ttl/bits/type_encoder.hpp>

TEST(std_encoding_test, test_std_encoding)
{
    using encoder =
        ttl::internal::basic_type_encoder<ttl::experimental::std_encoding>;
    using V = encoder::value_type;

    ASSERT_EQ(encoder::value<uint8_t>(), static_cast<V>(264));
    ASSERT_EQ(encoder::value<uint16_t>(), static_cast<V>(520));
    ASSERT_EQ(encoder::value<uint32_t>(), static_cast<V>(1032));
    ASSERT_EQ(encoder::value<uint64_t>(), static_cast<V>(2056));

    ASSERT_EQ(encoder::value<int8_t>(), static_cast<V>(65800));
    ASSERT_EQ(encoder::value<int16_t>(), static_cast<V>(66056));
    ASSERT_EQ(encoder::value<int32_t>(), static_cast<V>(66568));
    ASSERT_EQ(encoder::value<int64_t>(), static_cast<V>(67592));

    ASSERT_EQ(encoder::value<float>(), static_cast<V>(197640));
    ASSERT_EQ(encoder::value<double>(), static_cast<V>(198664));
    // ASSERT_EQ(encoder::value<long double>(), static_cast<V>(200712));
}

#include <ttl/bits/raw_tensor.hpp>
#include <ttl/bits/std_host_allocator.hpp>
#include <ttl/tensor>

using encoder =
    ttl::internal::basic_type_encoder<ttl::experimental::std_encoding>;
using cpu = ttl::internal::host_memory;
using raw_tensor = ttl::internal::raw_tensor<encoder, cpu>;
using raw_tensor_ref = ttl::internal::raw_tensor_ref<encoder, cpu>;
// using raw_tensor_view = ttl::internal::raw_tensor_view<encoder, cpu>;

template <typename R, ttl::rank_t r>
raw_tensor make_raw_tensor(const ttl::shape<r> &shape)
{
    return raw_tensor(raw_tensor::encoder_type::value<R>(), shape);
}

TEST(std_encoding_test, test_std_encoding_raw_tensor)
{
    {
        ttl::tensor<int, 1> x(1);
        raw_tensor_ref rx(ttl::ref(x));
    }
    {
        auto rt = make_raw_tensor<int>(ttl::make_shape(1, 2, 3));
    }
}
