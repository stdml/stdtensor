#pragma once
#include <ttl/bits/std_device.hpp>
#include <ttl/bits/std_tensor_fwd.hpp>
#include <ttl/bits/std_tensor_traits.hpp>

namespace ttl
{
namespace internal
{
template <typename R, typename S, typename D, typename A>
class basic_scalar_mixin
{
    using trait = basic_tensor_traits<R, A, D>;
    using data_ptr = typename trait::ptr_type;
    using data_ref = typename trait::ref_type;
    using data_t = typename trait::Data;

    data_t data_;

  protected:
    basic_scalar_mixin(data_ptr data) : data_(data)
    {
        static_assert(rank == 0, "");
    }

  public:
    using value_type = R;
    using shape_type = S;

    static constexpr auto rank = S::rank;  // == 0

    basic_scalar_mixin(data_ptr data, const S &) : data_(data) {}

    constexpr size_t data_size() const { return sizeof(R); }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + 1; }

    S shape() const { return S(); }

    void from_host(const void *data) const
    {
        basic_copier<D, host_memory>()(data_.get(), data, data_size());
    }

    void to_host(void *data) const
    {
        basic_copier<host_memory, D>()(data, data_.get(), data_size());
    }
};

template <typename R, typename S, typename D, typename A>
class basic_tensor_iterator
{
    using trait = basic_tensor_traits<R, A, D>;
    using data_ptr = typename trait::ptr_type;

    const S shape_;
    const size_t step_;

    data_ptr pos_;

    using T = basic_tensor<R, S, D, A>;

  public:
    basic_tensor_iterator(data_ptr pos, const S &shape)
        : shape_(shape), step_(shape.size()), pos_(pos)
    {
    }

    bool operator!=(const basic_tensor_iterator &it) const
    {
        return pos_ != it.pos_;
    }

    void operator++() { pos_ += step_; }

    T operator*() const { return T(pos_, shape_); }
};

template <typename R, typename S, typename D, typename A>
class basic_tensor_mixin
{
  protected:
    using trait = basic_tensor_traits<R, A, D>;
    using data_ptr = typename trait::ptr_type;
    using data_ref = typename trait::ref_type;
    using data_t = typename trait::Data;

    using allocator = basic_allocator<R, D>;

    using sub_shape = typename S::template subshape_t<1>;
    using element_t = basic_tensor<R, sub_shape, D, typename trait::Access>;
    using iterator =
        basic_tensor_iterator<R, sub_shape, D, typename trait::Access>;

    const S shape_;
    data_t data_;

    using index_type = typename S::dimension_type;

    iterator _iter(data_ptr pos) const
    {
        return iterator(pos, shape_.subshape());
    }

    explicit basic_tensor_mixin(data_ptr data, const S &shape)
        : shape_(shape), data_(data)
    {
        static_assert(rank > 0, "");
    }

  public:
    using value_type = R;
    using shape_type = S;

    using slice_type = basic_tensor<R, S, D, typename trait::Access>;

    static constexpr auto rank = S::rank;

    size_t data_size() const { return shape_.size() * sizeof(R); }

    const S &shape() const { return shape_; }

    data_ptr data() const { return data_.get(); }

    data_ptr data_end() const { return data_.get() + shape_.size(); }

    template <typename... I> data_ptr data(I... i) const
    {
        return data_.get() + shape_.offset(i...);
    }

    template <typename... I> data_ref at(I... i) const
    {  // FIXME: support other devices
        static_assert(std::is_same<D, host_memory>::value, "");
        return data_.get()[shape_.offset(i...)];
    }

    iterator begin() const { return _iter(data()); }

    iterator end() const { return _iter(data_end()); }

    element_t operator[](index_type i) const
    {
        return element_t(data_.get() + i * shape_.subspace_size(),
                         shape_.subshape());
    }

    slice_type slice(index_type i, index_type j) const
    {
        const auto sub_shape = shape_.subshape();
        return slice_type(data_.get() + i * sub_shape.size(),
                          batch(j - i, sub_shape));
    }

    void from_host(const void *data) const
    {
        basic_copier<D, host_memory>()(data_.get(), data, data_size());
    }

    void to_host(void *data) const
    {
        basic_copier<host_memory, D>()(data, data_.get(), data_size());
    }
};
}  // namespace internal
}  // namespace ttl
