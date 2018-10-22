#include <gtest/gtest.h>

#include <tensor.h>

#include <bits/std_scalar_type_encoding.hpp>

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

    ASSERT_EQ(ttl::internal::typeinfo<uint8_t>().code, dtypes.u8);
    ASSERT_EQ(ttl::internal::typeinfo<int8_t>().code, dtypes.i8);
    ASSERT_EQ(ttl::internal::typeinfo<int16_t>().code, dtypes.i16);
    ASSERT_EQ(ttl::internal::typeinfo<int32_t>().code, dtypes.i32);
    ASSERT_EQ(ttl::internal::typeinfo<float>().code, dtypes.f32);
    ASSERT_EQ(ttl::internal::typeinfo<double>().code, dtypes.f64);
}
