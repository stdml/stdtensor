#pragma once
#include <memory>

#include <ttl/bits/std_allocator.hpp>
#include <ttl/bits/std_base_tensor.hpp>
#include <ttl/bits/std_shape.hpp>

namespace ttl
{
namespace internal
{

/* forward declarations */

template <typename R, rank_t r, typename shape_t> class basic_tensor;
template <typename R, rank_t r, typename shape_t> class basic_tensor_ref;
template <typename R, rank_t r, typename shape_t> class basic_tensor_view;

/* specialization for rank 0 */

template <typename R, typename shape_t>
class basic_tensor_ref<R, 0, shape_t>
    : public base_scalar<R, shape_t, readwrite>
{
    using parent = base_scalar<R, shape_t, readwrite>;
    using parent::parent;

  public:
    using parent::data;

    basic_tensor_ref(const basic_tensor<R, 0, shape_t> &t) : parent(t.data()) {}

    R operator=(const R &val) const { return *data() = val; }
};

template <typename R, typename shape_t>
class basic_tensor_view<R, 0, shape_t>
    : public base_scalar<R, shape_t, readonly>
{
    using parent = base_scalar<R, shape_t, readonly>;
    using parent::parent;

  public:
    basic_tensor_view(const basic_tensor<R, 0, shape_t> &t) : parent(t.data())
    {
    }

    basic_tensor_view(const basic_tensor_ref<R, 0, shape_t> &t)
        : parent(t.data())
    {
    }
};

template <typename R, typename shape_t>
class basic_tensor<R, 0, shape_t> : public base_scalar<R, shape_t, owner>
{
    using parent = base_scalar<R, shape_t, owner>;

    std::unique_ptr<R> data_owner_;

    basic_tensor(R *data) : parent(data), data_owner_(data) {}

  public:
    using parent::data;

    basic_tensor() : basic_tensor(new R) {}

    explicit basic_tensor(const shape_t &_) : basic_tensor(new R) {}

    R operator=(const R &val) const { return *data() = val; }
};

template <typename R, typename shape_t>
R &scalar(const basic_tensor_ref<R, 0, shape_t> &t)
{
    return (t.data())[0];
}

template <typename R, typename shape_t>
R scalar(const basic_tensor_view<R, 0, shape_t> &t)
{
    return t.data()[0];
}

/* rank > 0 */

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_ref
    : public base_tensor<R, shape_t, readwrite, basic_tensor_ref>
{
    using parent = base_tensor<R, shape_t, readwrite, basic_tensor_ref>;

  public:
    template <typename... D>
    constexpr explicit basic_tensor_ref(R *data, D... d)
        : basic_tensor_ref(data, shape_t(d...))
    {
    }

    basic_tensor_ref(const basic_tensor<R, r, shape_t> &t)
        : basic_tensor_ref(t.data(), t.shape())
    {
    }

    constexpr explicit basic_tensor_ref(R *data, const shape_t &shape)
        : parent(data, shape)
    {
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor_view
    : public base_tensor<R, shape_t, readonly, basic_tensor_view>
{
    using parent = base_tensor<R, shape_t, readonly, basic_tensor_view>;

  public:
    template <typename... D>
    constexpr explicit basic_tensor_view(const R *data, D... d)
        : basic_tensor_view(data, shape_t(d...))
    {
    }

    basic_tensor_view(const basic_tensor<R, r, shape_t> &t)
        : basic_tensor_view(t.data(), t.shape())
    {
    }

    basic_tensor_view(const basic_tensor_ref<R, r, shape_t> &t)
        : basic_tensor_view(t.data(), t.shape())
    {
    }

    constexpr explicit basic_tensor_view(const R *data, const shape_t &shape)
        : parent(data, shape)
    {
    }
};

template <typename R, rank_t r, typename shape_t = basic_shape<r>>
class basic_tensor : public base_tensor<R, shape_t, readwrite, basic_tensor_ref>
{
    using allocator = basic_allocator<R>;
    using owner = std::unique_ptr<R[]>;

    using parent = base_tensor<R, shape_t, readwrite, basic_tensor_ref>;

    owner data_owner_;

    explicit basic_tensor(R *data, const shape_t &shape)
        : parent(data, shape), data_owner_(data)
    {
    }

  public:
    template <typename... D>
    constexpr explicit basic_tensor(D... d) : basic_tensor(shape_t(d...))
    {
    }

    constexpr explicit basic_tensor(const shape_t &shape)
        : basic_tensor(allocator()(shape.size()), shape)
    {
    }
};
}  // namespace internal
}  // namespace ttl
