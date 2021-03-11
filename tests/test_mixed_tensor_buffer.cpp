#include "testing.hpp"

#include <ttl/bits/mixed_tensor_buffer.hpp>
#include <ttl/debug>

namespace ttl
{
template <typename E, typename D = ttl::host_memory>
using mixed_tensor_buffer = internal::basic_mixed_tensor_buffer<E, D>;

using idx_encoder =
    ttl::internal::basic_type_encoder<ttl::internal::idx_format::encoding>;
}  // namespace ttl

TEST(tensor_buffer_test, test2)
{
    using tensor_buffer = ttl::mixed_tensor_buffer<ttl::idx_encoder>;
    using S = typename tensor_buffer::symbol_type;
    std::vector<S> symbols;
    symbols.emplace_back(S::type<float>(), ttl::flat_shape(28, 28, 10));
    symbols.emplace_back(S::type<int8_t>(), ttl::flat_shape(10));
    tensor_buffer tb(symbols);
    ASSERT_EQ(static_cast<int>(tb.size()), 2);
    ASSERT_EQ(static_cast<size_t>(28 * 28 * 10 * 4 + 10), tb.data_size());
    tb[0].typed<float, 3>();
    tb[1].typed<int8_t, 1>();
    for (const auto &t : tb) {
        printf("%s\n", ttl::to_string(t.shape()).c_str());
    }
}
