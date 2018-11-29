#include "testing.hpp"

#include <ttl/tensor>

template <typename Dim, ttl::internal::rank_t r> void test_dimemsion_type()
{
    {
        using shape_t = ttl::internal::basic_shape<r, Dim>;
        static_assert(
            std::is_same<typename shape_t::dimension_type, Dim>::value, "");
    }
    {
        using shape_t = ttl::internal::basic_strided_shape<r, Dim>;
        static_assert(
            std::is_same<typename shape_t::dimension_type, Dim>::value, "");
    }
}

TEST(shape_concept_test, concept_test)
{
    test_dimemsion_type<int, 0>();
    test_dimemsion_type<int, 1>();
    test_dimemsion_type<int, 2>();
}
