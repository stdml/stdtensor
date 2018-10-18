#include <cstdarg>

#include <tensor.h>

#include <stdtensor>

#include <bits/std_generic_tensor.hpp>

template <typename T>
using encoding = ttl::internal::default_data_type_encoding<T>;

using generic_shape = ttl::internal::basic_generic_shape<std::uint32_t>;
using generic_tensor = ttl::internal::basic_generic_tensor<generic_shape>;

class tensor_s
{
  public:
    tensor_s(generic_tensor *ptr) : ptr(ptr) {}

  private:
    std::unique_ptr<generic_tensor> ptr;
};

tensor_t *new_tensor(uint8_t data_type, int rank, ...)
{
    std::vector<uint32_t> dims;
    va_list list;
    va_start(list, rank);
    for (auto i = 0; i < rank; ++i) {
        uint32_t dim = va_arg(list, uint32_t);
        dims.push_back(dim);
    }
    va_end(list);
    generic_shape shape(dims);

    ttl::internal::data_type_info value_type = {data_type, 1};
    return new tensor_s(new generic_tensor(value_type, shape));
}

void del_tensor(const tensor_t *pt) { delete pt; }
