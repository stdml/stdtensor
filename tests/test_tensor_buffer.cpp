#include "testing.hpp"

#include <ttl/experimental/tensor_buffer>

namespace ttl
{
using namespace experimental;
}

TEST(tensor_buffer_test, test_1)
{
    using S = typename ttl::tensor_buffer<int>::shape_type;
    std::vector<S> shapes;
    shapes.emplace_back(ttl::make_shape(28, 28, 10));
    shapes.push_back(S(ttl::make_shape(10)));
    ttl::tensor_buffer<int> tb(shapes);
    ASSERT_EQ(static_cast<int>(tb.size()), 2);
    tb[0].ranked<3>();
    tb[1].ranked<1>();
}

TEST(tensor_buffer_test, test2)
{
    using E =
        ttl::internal::basic_type_encoder<ttl::internal::idx_format::encoding>;
    using mtb = ttl::mixed_tensor_buffer<E>;
    using S = typename mtb::symbol_type;

    std::vector<S> symbols;
    symbols.emplace_back(S::type<float>(), ttl::flat_shape(28, 28, 10));
    symbols.emplace_back(S::type<int8_t>(), ttl::flat_shape(10));
    mtb tb(symbols);
    ASSERT_EQ(static_cast<int>(tb.size()), 2);
    ASSERT_EQ(static_cast<size_t>(28 * 28 * 10 * 4 + 10), tb.data_size());
    tb[0].typed<float, 3>();
    tb[1].typed<int8_t, 1>();

    std::vector<size_t> sizes = {28 * 28 * 10, 10};
    int i = 0;
    for (const auto &t : tb) {
        ASSERT_EQ(t.size(), sizes[i]);
        ++i;
    }
    ASSERT_EQ(i, 2);
}
