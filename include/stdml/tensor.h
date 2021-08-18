#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum stdml_device_e {
    cpu,
    gpu,
};

typedef enum stdml_device_e stdml_device_t;

enum stdml_dtype_e {
    u8,
    u16,
    u32,
    u64,

    i8,
    i16,
    i32,
    i64,

    bf16,
    f16,
    f32,
    f64,

    boolean,
    str,
    resource
};

typedef enum stdml_dtype_e stdml_dtype_t;

// typedef struct shape_s shape_t;
typedef struct tensor_s tensor_t;

extern tensor_t *stdml_new_tensor(stdml_dtype_t dt, stdml_device_t dev,
                                  int rank, ...);

extern void stdml_del_tensor(tensor_t * /*! p_tensor_t */);

#ifdef __cplusplus
}
#endif
