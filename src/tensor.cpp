#include <cstdarg>

#include <stdtensor>
#include <tensor.h>

using ttl::generic_shape;
using ttl::generic_tensor;

struct tensor_s {
    tensor_s(generic_tensor *ptr) : ptr_(ptr) {}

  private:
    std::unique_ptr<generic_tensor> ptr_;
};

tensor_t *new_tensor(uint8_t data_type, int rank, ...)
{
    using dim_t = generic_shape::dimension_type;

    std::vector<dim_t> dims;
    va_list list;
    va_start(list, rank);
    for (auto i = 0; i < rank; ++i) {
        dim_t dim = va_arg(list, dim_t);
        dims.push_back(dim);
    }
    va_end(list);
    generic_shape shape(dims);

    ttl::internal::data_type_info value_type = {
        data_type, 1};  // FIXME: get data size by data_type
    return new tensor_s(new generic_tensor(value_type, shape));
}

void del_tensor(const tensor_t *pt) { delete pt; }
