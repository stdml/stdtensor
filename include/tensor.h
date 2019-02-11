#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dtypes_t dtypes_t;

struct dtypes_t {
    const uint8_t u8;
    const uint8_t i8;
    const uint8_t i16;
    const uint8_t i32;
    const uint8_t f32;
    const uint8_t f64;
};

extern const dtypes_t dtypes;

// typedef struct shape_s shape_t;
typedef struct tensor_s tensor_t;

extern tensor_t *new_tensor(uint8_t /*! value_type */, int /*! rank */, ...);

extern tensor_t *new_tensor1(uint8_t /*! value_type */, int /*! rank */,
                             const int * /* dims */);

extern void del_tensor(const tensor_t * /*! p_tensor_t */);

extern void *tensor_data(tensor_t * /*! p_tensor_t */);

#ifdef __cplusplus
}
#endif
