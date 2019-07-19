#pragma once

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D> class base_tensor
{
  public:
    using value_type = R;
    using shape_type = S;

  protected:
    using data_type = D;
    using data_ptr = typename D::ptr_type;
    using data_ref = typename D::ref_type;

    const shape_type shape_;
    data_type data_;

    using index_type = typename S::dimension_type;

  public:
    base_tensor(data_ptr data, const S &shape) : shape_(shape), data_(data)
    {
        // TODO: static_assert();
    }

    S shape() const { return shape_; }

    auto data() const { return data_.get(); }

    auto data_end() const { return data_.get() + shape_.size(); }

    template <typename... I> data_ref at(I... i) const
    {
        return data_.get()[shape_.offset(i...)];
    }

  protected:
    template <typename subspace_t> subspace_t _bracket(index_type i) const
    {
        return subspace_t(data_.get() + i * shape_.subspace_size(),
                          shape_.subshape());
    }

    template <typename self_t> self_t _slice(index_type i, index_type j) const
    {
        const auto sub_shape = shape_.subshape();
        return self_t(data_.get() + i * sub_shape.size(),
                      batch(j - i, sub_shape));
    }
};

}  // namespace internal
}  // namespace ttl
