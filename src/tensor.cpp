#include <cstdarg>

#include <tensor.h>
#include <ttl/experimental/raw_tensor>

using ttl::experimental::raw_tensor;
using scalar_encoding = raw_tensor::encoder_type;

const dtypes_t dtypes = {
    scalar_encoding::value<uint8_t>(),  //
    scalar_encoding::value<int8_t>(),   //
    scalar_encoding::value<int16_t>(),  //
    scalar_encoding::value<int32_t>(),  //
    scalar_encoding::value<float>(),    //
    scalar_encoding::value<double>(),   //
};

struct tensor_s : raw_tensor {
    using raw_tensor::raw_tensor;
};

tensor_t *new_tensor(uint8_t value_type, int rank, ...)
{
    using raw_shape = raw_tensor::shape_type;
    using dim_t = raw_shape::dimension_type;

    std::vector<dim_t> dims;
    va_list list;
    va_start(list, rank);
    for (auto i = 0; i < rank; ++i) {
        dim_t dim = va_arg(list, dim_t);
        dims.push_back(dim);
    }
    va_end(list);
    raw_shape shape(dims);

    return new tensor_s(value_type, shape);
}

void del_tensor(const tensor_t *pt) { delete pt; }
