#include <random>

#include "testing.hpp"

#include <ttl/experimental/sparse>
#include <ttl/tensor>

void test_sparse_matrix(int h, int w)
{
    using SPM = ttl::experimental::sparse_matrix<int>;
    using Mat = ttl::matrix<int>;

    Mat x(h, w);

    std::random_device device;
    std::mt19937 g(device());

    std::generate(x.data(), x.data_end(), [&] {
        auto v = g();
        return v % 10 == 0 ? v : 0;
    });

    auto triples = SPM::coo(x);

    SPM m = SPM::from_triples(h, w, triples);
    Mat y = m.dense();

    ASSERT_TRUE(std::equal(x.data(), x.data_end(), y.data()));
}

TEST(sparse_test, test_1)
{
    for_all_permutations(test_sparse_matrix, 2, 4);
    for_all_permutations(test_sparse_matrix, 3, 5);
    for_all_permutations(test_sparse_matrix, 10, 11);
    for_all_permutations(test_sparse_matrix, 99, 101);
}
