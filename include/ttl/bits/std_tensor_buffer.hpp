#pragma once
#include <vector>

#include <ttl/device>
#include <ttl/experimental/flat_tensor>
#include <ttl/flat_shape>
#include <ttl/tensor>

namespace ttl
{
namespace internal
{
using ttl::experimental::flat_tensor_ref;
using ttl::experimental::flat_tensor_view;

template <typename R, typename D>
class basic_tensor_buffer
{
    using shape_list = std::vector<flat_shape>;
    using index_t = std::vector<size_t>;

    static size_t total_size(const shape_list &shapes)
    {
        return std::accumulate(shapes.begin(), shapes.end(),
                               static_cast<size_t>(0),
                               [](size_t acc, const flat_shape &shape) {
                                   return acc + shape.size();
                               });
    }

    static std::vector<size_t> build_offsets(const shape_list &shapes)
    {
        std::vector<size_t> offsets;
        offsets.reserve(shapes.size());
        size_t off = 0;
        for (auto &shape : shapes) {
            offsets.push_back(off);
            off += shape.size();
        }
        return offsets;
    }

    ttl::vector<R, D> buffer;
    const index_t offsets;
    const shape_list shapes;

  public:
    using shape_type = flat_shape;

    basic_tensor_buffer(shape_list shapes)
        : buffer(total_size(shapes)), offsets(build_offsets(shapes)),
          shapes(std::move(shapes))
    {
    }

    R *data(int i) { return buffer.data() + offsets.at(i); }

    const R *data(int i) const { return buffer.data() + offsets.at(i); }

    flat_tensor_ref<R> ref(int i)
    {
        return flat_tensor_ref<R>(data(i), shapes.at(i));
    }

    flat_tensor_ref<R> operator[](int i) { return ref(i); }

    template <rank_t r>
    tensor_ref<R, r> ref(int i)
    {
        return tensor_ref<R, r>(data(i), shapes.at(i).template ranked<r>());
    }

    template <rank_t r>
    tensor_view<R, r> view(int i) const
    {
        return tensor_view<R, r>(data(i), shapes.at(i).template ranked<r>());
    }

    size_t size() const { return shapes.size(); }

    class iterator
    {
        basic_tensor_buffer *tb;
        int idx;

      public:
        iterator(basic_tensor_buffer *tb, int idx) : tb(tb), idx(idx) {}

        bool operator!=(const iterator &i) const { return idx != i.idx; }

        void operator++() { idx++; }

        flat_tensor_ref<R> operator*() { return tb->ref(idx); }
    };

    iterator begin() { return iterator(this, 0); }

    iterator end() { return iterator(this, shapes.size()); }
};
}  // namespace internal
}  // namespace ttl
