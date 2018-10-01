#pragma once
#include <memory>

#include <bits/std_shape.hpp>

namespace ttl
{
namespace internal
{

/* forward declarations */

template <typename R, rank_t r, typename shape_t, typename elem_t>
class basic_tensor_iterator;

template <typename R, rank_t r, typename shape_t> class basic_tensor;
template <typename R, rank_t r, typename shape_t> class basic_tensor_ref;
template <typename R, rank_t r, typename shape_t> class basic_tensor_view;

/* specialization for rank 0 */

template <typename R, typename shape_t, typename elem_t>
class basic_tensor_iterator<R, 0, shape_t, elem_t>
{
    const R *pos;

  public:
    basic_tensor_iterator(const R *pos, const shape_t & /* shape */) : pos(pos)
    {
    }

    bool operator!=(const basic_tensor_iterator &it) const
    {
        return pos != it.pos;
    }

    void operator++() { ++pos; }

    void _advance(size_t k) { pos += k; }

    elem_t operator*() const { return elem_t((R *)/* FIXME */ pos, shape_t()); }
};

template <typename R, typename shape_t> class basic_tensor_ref<R, 0, shape_t>
{
  public:
    basic_tensor_ref(R *data) : data_(data) {}

    basic_tensor_ref(R *data, const shape_t &) : data_(data) {}

    // R *data() { return data_; }
    // const R *data() const { return data_; }
    //   private:
    R *const data_;
};

template <typename R, typename shape_t> class basic_tensor_view<R, 0, shape_t>
{
  public:
    basic_tensor_view(const R *data) : data_(data) {}

    basic_tensor_view(const R *data, const shape_t &) : data_(data) {}

    const R *data() const { return data_; }

  private:
    const R *const data_;
};

template <typename R, typename shape_t>
R &scalar(const basic_tensor_ref<R, 0, shape_t> &t)
{
    return t.data_[0];
}

template <typename R, typename shape_t>
R scalar(const basic_tensor_view<R, 0, shape_t> &t)
{
    return t.data()[0];
}

template <template <typename, rank_t, typename> class T, typename R, rank_t r,
          typename shape_t>
basic_tensor_ref<R, r, shape_t> ref(const T<R, r, shape_t> &t)
{
    const R *const c_ptr = t.data();
    R *ptr = (R *)/* FIXME */ c_ptr;
    return basic_tensor_ref<R, r, shape_t>(ptr, t.shape());
}

/* rank > 0 */

template <typename R, rank_t r, typename shape_t, typename elem_t>
class basic_tensor_iterator
{
    const shape_t shape;
    const size_t step;

    const R *pos;

  public:
    basic_tensor_iterator(const R *pos, const shape_t &s)
        : shape(s), step(s.size()), pos(pos)
    {
    }

    bool operator!=(const basic_tensor_iterator &it) const
    {
        return pos != it.pos;
    }

    void operator++() { pos += step; }

    void _advance(size_t k) { pos += k * step; }

    elem_t operator*() const { return elem_t((R *)/* FIXME */ pos, shape); }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_ref
{
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_tensor_ref<R, r - 1, subshape_shape_t>;
    using iterator =
        basic_tensor_iterator<R, r - 1, subshape_shape_t, subspace_t>;

    const shape_t shape_;
    R *const data_;

  public:
    template <typename... D>
    constexpr explicit basic_tensor_ref(R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    constexpr explicit basic_tensor_ref(R *data, const shape_t &shape)
        : shape_(shape), data_(data)
    {
    }

    R *data() { return data_; }

    const R *data() const { return data_; }

    shape_t shape() const { return shape_; }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
    }

    iterator begin() const { return iterator(data_, shape_.subshape()); }

    iterator end() const
    {
        return iterator(data_ + shape_.size(), shape_.subshape());
    }

    subspace_t operator[](int i) const
    {
        return subspace_t(data_ + i * shape_.subspace_size(),
                          shape_.subshape());
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_view
{
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_t = basic_tensor_view<R, r - 1, subshape_shape_t>;
    using iterator =
        basic_tensor_iterator<R, r - 1, subshape_shape_t, subspace_t>;

    const shape_t shape_;
    const R *const data_;

  public:
    template <typename... D>
    constexpr explicit basic_tensor_view(const R *data, D... d)
        : shape_(d...), data_(data)
    {
    }

    constexpr explicit basic_tensor_view(const R *data, const shape_t &shape)
        : shape_(shape), data_(data)
    {
    }

    const R *data() const { return data_; }

    shape_t shape() const { return shape_; }

    template <typename... I> R at(I... i) { return data_[shape_.offset(i...)]; }

    iterator begin() const { return iterator(data_, shape_.subshape()); }

    iterator end() const
    {
        return iterator(data_ + shape_.size(), shape_.subshape());
    }

    subspace_t operator[](int i) const
    {
        return subspace_t(data_ + i * shape_.subspace_size(),
                          shape_.subshape());
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor
{
    using subshape_shape_t = typename shape_t::template subshape_t<1>;
    using subspace_ref_t = basic_tensor_ref<R, r - 1, subshape_shape_t>;
    using iterator =
        basic_tensor_iterator<R, r - 1, subshape_shape_t, subspace_ref_t>;

    const shape_t shape_;
    const std::unique_ptr<R[]> data_;

  public:
    template <typename... D>
    constexpr explicit basic_tensor(D... d)
        : shape_(d...), data_(new R[shape_.size()])
    {
    }

    R *data() { return data_.get(); }

    const R *data() const { return data_.get(); }

    shape_t shape() const { return shape_; }

    template <typename... I> R &at(I... i)
    {
        return data_[shape_.offset(i...)];
    }

    subspace_ref_t operator[](int i) const
    {
        return subspace_ref_t(data_.get() + i * shape_.subspace_size(),
                              shape_.subshape());
    }

    iterator begin() const { return ref(*this).begin(); }

    iterator end() const { return ref(*this).end(); }
};
}  // namespace internal
}  // namespace ttl
