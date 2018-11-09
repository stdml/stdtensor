#include <gtest/gtest.h>

#include <stdtensor>
#include <tensor.h>

using ttl::raw_tensor;

TEST(c_api_test, test1)
{
    for (auto dt : {dtypes.u8, dtypes.i8, dtypes.i16, dtypes.i32, dtypes.f32,
                    dtypes.f64}) {
        {
            tensor_t *pt = new_tensor(dt, 1, 1);
            del_tensor(pt);
        }
        {
            tensor_t *pt = new_tensor(dt, 2, 2, 3);
            del_tensor(pt);
        }
    }

    using scalar_encoding = raw_tensor::encoder_type;

    ASSERT_EQ(scalar_encoding::value<uint8_t>(), dtypes.u8);
    ASSERT_EQ(scalar_encoding::value<int8_t>(), dtypes.i8);
    ASSERT_EQ(scalar_encoding::value<int16_t>(), dtypes.i16);
    ASSERT_EQ(scalar_encoding::value<int32_t>(), dtypes.i32);
    ASSERT_EQ(scalar_encoding::value<float>(), dtypes.f32);
    ASSERT_EQ(scalar_encoding::value<double>(), dtypes.f64);
}
