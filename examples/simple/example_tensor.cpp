#include <iostream>
#include <ttl/range>
#include <ttl/tensor>

void transpose(const ttl::tensor_view<int, 2> &x,
               const ttl::tensor_ref<int, 2> &y)
{
    for (auto i : ttl::range<0>(x)) {
        for (auto j : ttl::range<1>(x)) { y[j][i] = x[i][j]; }
    }
}

int main()
{
    ttl::tensor<int, 2> m(2, 3);
    int idx = 0;
    for (const auto &row : m) {
        for (int &x : row) { x = idx++; }
    }
    ttl::tensor<int, 2> n(3, 2);
    transpose(ttl::view(m), ttl::ref(n));
    for (const auto &row : n) {
        for (int x : row) { std::cout << x << " "; }
        std::cout << std::endl;
    }
    return 0;
}
