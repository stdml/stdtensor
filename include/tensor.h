#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// typedef struct shape_s shape_t;
typedef struct tensor_s tensor_t;

extern tensor_t *new_tensor(uint8_t /*! data_type */, int /*! rank */, ...);

extern void del_tensor(const tensor_t * /*! p_tensor_t */);

#ifdef __cplusplus
}
#endif
