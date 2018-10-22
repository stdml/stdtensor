#include <cstdarg>

#include <stdtensor>
#include <tensor.h>

using ttl::raw_shape;
using ttl::raw_tensor;

template <typename R>
using scalar_encoding = ttl::internal::default_scalar_type_encoding<R>;

const dtypes_t dtypes = {
    scalar_encoding<uint8_t>::value,  //
    scalar_encoding<int8_t>::value,   //
    scalar_encoding<int16_t>::value,  //
    scalar_encoding<int32_t>::value,  //
    scalar_encoding<float>::value,    //
    scalar_encoding<double>::value,   //
};

struct tensor_s : raw_tensor {
    using raw_tensor::raw_tensor;
};

tensor_t *new_tensor(uint8_t data_type, int rank, ...)
{
    const auto info = ttl::internal::default_scalar_info_t(data_type);

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

    ttl::internal::data_type_info value_type = {
        data_type, static_cast<uint8_t>(info.size())};
    return new tensor_s(value_type, shape);
}

void del_tensor(const tensor_t *pt) { delete pt; }
