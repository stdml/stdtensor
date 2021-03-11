#pragma once
#include <vector>

#include <ttl/device>
#include <ttl/experimental/raw_tensor>
#include <ttl/experimental/type>
#include <ttl/flat_shape>
#include <ttl/tensor>

namespace ttl
{
namespace internal
{
template <typename E, typename D>
class basic_mixed_tensor_buffer
{
    using symbol_t = ttl::experimental::raw_type<E>;
    using symbol_list = std::vector<symbol_t>;
    using index_t = std::vector<size_t>;

    static size_t total_data_size(const symbol_list &ss)
    {
        return std::accumulate(
            ss.begin(), ss.end(), static_cast<size_t>(0),
            [](size_t acc, const symbol_t &s) { return acc + s.data_size(); });
    }

    static std::vector<size_t> build_offsets(const symbol_list &ss)
    {
        std::vector<size_t> offsets;
        offsets.reserve(ss.size());
        size_t off = 0;
        for (auto &s : ss) {
            offsets.push_back(off);
            off += s.data_size();
        }
        return offsets;
    }

    ttl::vector<char, D> buffer;
    const index_t offsets;
    const symbol_list symbols;

  public:
    using symbol_type = symbol_t;

    basic_mixed_tensor_buffer(symbol_list symbols)
        : buffer(total_data_size(symbols)), offsets(build_offsets(symbols)),
          symbols(std::move(symbols))
    {
    }

    size_t size() const { return symbols.size(); }

    size_t data_size() const { return buffer.data_size(); }

    void *data(int i) { return buffer.data() + offsets.at(i); }

    template <typename R>
    R *data(int i)
    {
        return symbols.at(i).template typed<R>(data(i));
    }

    raw_tensor_ref<E, D> ref(int i)
    {
        const symbol_t &symbol = symbols.at(i);
        return raw_tensor_ref<E, D>(data(i), symbol.value_type(),
                                    symbol.shape());
    }

    template <typename R, rank_t r>
    tensor_ref<R, r, D> ref(int i)
    {
        return ref(i).template typed<R, r>();
    }

    raw_tensor_ref<E, D> operator[](int i) { return ref(i); }

    class iterator
    {
        basic_mixed_tensor_buffer *tb;
        int idx;

      public:
        iterator(basic_mixed_tensor_buffer *tb, int idx) : tb(tb), idx(idx) {}

        bool operator!=(const iterator &i) const { return idx != i.idx; }

        void operator++() { idx++; }

        raw_tensor_ref<E, D> operator*() { return tb->ref(idx); }
    };

    iterator begin() { return iterator(this, 0); }

    iterator end() { return iterator(this, symbols.size()); }
};
}  // namespace internal
}  // namespace ttl
