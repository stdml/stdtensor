#include <stdml/bits/tensor/tensor.hpp>
#include <stdml/tensor.h>

struct tensor_s {
    stdml::Tensor t;
};

stdml::Device from(stdml_device_t dev)
{
    switch (dev) {
    case cpu:
        return stdml::cpu;
    case gpu:
        return stdml::gpu;
    default:
        throw std::invalid_argument(__func__);
    }
}

stdml::DType from(stdml_dtype_t dt)
{
    switch (dt) {
    case u8:
        return stdml::u8;
    case u16:
        return stdml::u16;
    case u32:
        return stdml::u32;
    case u64:
        return stdml::u64;
        //
    case i8:
        return stdml::i8;
    case i16:
        return stdml::i16;
    case i32:
        return stdml::i32;
    case i64:
        return stdml::i64;
        //
    case f32:
        return stdml::f32;
    case f64:
        return stdml::f64;
    default:
        throw std::invalid_argument(__func__);
    }
}

tensor_t *stdml_new_tensor(stdml_dtype_t dt, stdml_device_t dev, int rank, ...)
{
    std::vector<int> dims;
    va_list list;
    va_start(list, rank);
    for (auto i = 0; i < rank; ++i) {
        int dim = va_arg(list, int);
        dims.push_back(dim);
    }
    va_end(list);

    return new tensor_s{
        stdml::Tensor(from(dt), stdml::Shape(std::move(dims)), from(dev))};
}

void stdml_del_tensor(tensor_t *p) { delete p; }
