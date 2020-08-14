#pragma once
#include <algorithm>
#include <tuple>
#include <vector>

#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_host_tensor.hpp>
#include <ttl/bits/std_sparse_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename I>
void compress_row_idx(const std::vector<I> &sorted_row_idx,
                      std::vector<I> &row_ptr)
{
    const I n = row_ptr.size();
    const I m = sorted_row_idx.size();
    I j = 0;
    for (I i = 0; i < n; ++i) {
        row_ptr[i] = j;
        for (; j < m; ++j) {
            if (sorted_row_idx[j] != i) { break; }
        }
    }
    if (j != m) { throw std::invalid_argument("row_idx is not sorted"); }
}

template <typename R, typename I>
class basic_sparse_tensor<R, 2, I, host_memory>
{
    using triple = std::tuple<I, I, R>;

    const I n_rows_;
    const I n_cols_;

    std::vector<I> row_ptr_;
    std::vector<I> col_idx_;
    std::vector<R> values_;

    basic_sparse_tensor(I l, std::vector<I> row_ptr, std::vector<I> col_idx,
                        std::vector<R> values)
        : n_rows_(row_ptr.size()),
          n_cols_(l),
          row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)),
          values_(std::move(values))
    {
    }

    using dense_t = basic_host_tensor<R, 2, I>;
    using coordinate_list_t = std::vector<triple>;

  public:
    static basic_sparse_tensor from_triples(I n, I l, coordinate_list_t triples,
                                            bool sorted = false)
    {
        if (!sorted) { std::sort(triples.begin(), triples.end()); }
        const I m = triples.size();

        std::vector<I> row_idx(m);
        std::vector<I> col_idx(m);
        std::vector<R> values(m);

        std::transform(triples.begin(), triples.end(), row_idx.begin(),
                       [](auto &t) { return std::get<0>(t); });
        std::transform(triples.begin(), triples.end(), col_idx.begin(),
                       [](auto &t) { return std::get<1>(t); });
        std::transform(triples.begin(), triples.end(), values.begin(),
                       [](auto &t) { return std::get<2>(t); });

        std::vector<I> row_ptr(n);
        compress_row_idx(row_idx, row_ptr);

        return basic_sparse_tensor(l, std::move(row_ptr), std::move(col_idx),
                                   std::move(values));
    }

    static coordinate_list_t coo(const dense_t &x)
    {
        static_assert(std::is_integral<R>::value, "");

        coordinate_list_t triples;
        I n = std::get<0>(x.dims());
        I l = std::get<1>(x.dims());
        for (I i = 0; i < n; ++i) {
            for (I j = 0; j < l; ++j) {
                R v = x.at(i, j);
                if (v != 0) { triples.push_back(std::make_tuple(i, j, v)); }
            }
        }
        return triples;
    }

    dense_t dense() const
    {
        dense_t x(n_rows_, n_cols_);
        std::fill(x.data(), x.data_end(), static_cast<R>(0));
        for (I i = 0; i < n_rows_; ++i) {
            const I begin = row_ptr_[i];
            const I end = i + 1 < n_rows_ ? row_ptr_[i + 1] : values_.size();
            for (I j = begin; j < end; ++j) {
                x.at(i, col_idx_[j]) = values_[j];
            }
        }
        return x;
    }
};
}  // namespace internal
}  // namespace ttl
