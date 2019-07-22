#pragma once

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D> class base_scalar
{
  public:
    using value_type = R;
    using shape_type = S;

    static constexpr auto rank = S::rank;

  protected:
    using data_ptr = typename D::ptr_type;
    using data_ref = typename D::ref_type;

    D data_;

  public:
    base_scalar(data_ptr data) : data_(data) { static_assert(rank == 0, ""); }

    base_scalar(data_ptr data, const S &) : data_(data) {}

    size_t data_size() const { return sizeof(R); }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + 1; }

    S shape() const { return S(); }
};

template <typename R, typename S, typename D, typename T>
class base_tensor_iterator
{
    using data_ptr = typename D::ptr_type;

    const S shape_;
    const size_t step_;

    data_ptr pos_;

  public:
    base_tensor_iterator(data_ptr pos, const S &shape)
        : shape_(shape), step_(shape.size()), pos_(pos)
    {
    }

    bool operator!=(const base_tensor_iterator &it) const
    {
        return pos_ != it.pos_;
    }

    void operator++() { pos_ += step_; }

    T operator*() const { return T(pos_, shape_); }
};

using rank_t = uint8_t;

template <typename R, typename S, typename D,
          template <typename, rank_t, typename> class T>
class base_tensor
{
  public:
    using value_type = R;
    using shape_type = S;

    static constexpr auto rank = S::rank;

  protected:
    using sub_shape = typename S::template subshape_t<1>;
    using slice_t = T<R, rank, S>;
    using element_t = T<R, rank - 1, sub_shape>;
    using iterator = base_tensor_iterator<R, sub_shape, D, element_t>;

    using data_ptr = typename D::ptr_type;
    using data_ref = typename D::ref_type;

    const S shape_;
    D data_;

    using index_type = typename S::dimension_type;

    iterator _iter(data_ptr pos) const
    {
        return iterator(pos, shape_.subshape());
    }

  public:
    base_tensor(data_ptr data, const S &shape) : shape_(shape), data_(data)
    {
        static_assert(rank > 0, "");
    }

    size_t data_size() const { return shape_.size() * sizeof(R); }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + shape_.size(); }

    template <typename... I> data_ptr data(I... i) const
    {
        return data_.get() + shape_.offset(i...);
    }

    template <typename... I> data_ref at(I... i) const
    {
        return data_.get()[shape_.offset(i...)];
    }

    iterator begin() const { return _iter(data()); }

    iterator end() const { return _iter(data_end()); }

    element_t operator[](index_type i) const
    {
        return element_t(data_.get() + i * shape_.subspace_size(),
                         shape_.subshape());
    }

    slice_t slice(index_type i, index_type j) const
    {
        const auto sub_shape = shape_.subshape();
        return slice_t(data_.get() + i * sub_shape.size(),
                       batch(j - i, sub_shape));
    }
};

}  // namespace internal
}  // namespace ttl
