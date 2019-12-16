#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <ttl/bits/flat_tensor.hpp>
#include <ttl/bits/raw_shape.hpp>
#include <ttl/bits/raw_tensor_mixin.hpp>
#include <ttl/bits/std_host_tensor.hpp>

namespace ttl
{
namespace internal
{
template <typename DataEncoder, typename Dim = uint32_t>
class basic_raw_tensor_own;

template <typename DataEncoder, typename Dim = uint32_t>
class basic_raw_tensor_ref;

template <typename DataEncoder, typename Dim = uint32_t>
class basic_raw_tensor_view;

template <typename DataEncoder, typename Dim>
class basic_raw_tensor_own
    : public raw_tensor_mixin<DataEncoder, basic_raw_shape<Dim>, host_memory,
                              owner>
{
    using value_type_t = typename DataEncoder::value_type;
    using S = basic_raw_shape<Dim>;
    using mixin = raw_tensor_mixin<DataEncoder, S, host_memory, owner>;
    using allocator = basic_allocator<char, host_memory>;

  public:
    template <typename... Dims>
    explicit basic_raw_tensor_own(const value_type_t value_type, Dims... d)
        : basic_raw_tensor_own(value_type, S(d...))
    {
    }

    explicit basic_raw_tensor_own(const value_type_t value_type, const S &shape)
        : mixin(allocator()(shape.size() * DataEncoder::size(value_type)),
                value_type, shape)
    {
    }

    template <typename R>
    R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != this->value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<R *>(this->data_.get());
    }

    void *data() const { return this->data_.get(); }

    void *data_end() const
    {
        return static_cast<char *>(this->data_.get()) + this->data_size();
    }

    template <typename R, rank_t r>
    basic_host_tensor_ref<R, r, Dim> ref_as() const
    {
        return ranked_as<basic_host_tensor_ref<R, r, Dim>>();
    }

    template <typename R, rank_t r>
    basic_host_tensor_view<R, r, Dim> view_as() const
    {
        return ranked_as<basic_host_tensor_view<R, r, Dim>>();
    }

    template <typename R>
    basic_flat_tensor_ref<R, S> typed_as() const
    {
        return basic_flat_tensor_ref<R, S>(data<R>(), this->shape_);
    }

  private:
    template <typename T>
    T ranked_as() const
    {
        return T(data<typename T::value_type>(),
                 this->shape_.template as_ranked<T::rank>());
    }
};

template <typename DataEncoder, typename Dim>
class basic_raw_tensor_ref
    : public raw_tensor_mixin<DataEncoder, basic_raw_shape<Dim>, host_memory,
                              readwrite>
{
    using value_type_t = typename DataEncoder::value_type;
    using S = basic_raw_shape<Dim>;
    using mixin = raw_tensor_mixin<DataEncoder, S, host_memory, readwrite>;

    template <typename R>
    R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != this->value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<R *>(this->data_.get());
    }

  public:
    template <typename... Dims>
    explicit basic_raw_tensor_ref(void *data, const value_type_t value_type,
                                  Dims... d)
        : basic_raw_tensor_ref(data, value_type, S(d...))
    {
    }

    explicit basic_raw_tensor_ref(void *data, const value_type_t value_type,
                                  const S &shape)
        : mixin(data, value_type, shape)
    {
    }

    explicit basic_raw_tensor_ref(
        const basic_raw_tensor_own<DataEncoder, Dim> &t)
        : basic_raw_tensor_ref(t.data(), t.value_type(), t.shape())
    {
    }

    template <typename R, rank_t r>
    explicit basic_raw_tensor_ref(const basic_host_tensor_ref<R, r, Dim> &t)
        : basic_raw_tensor_ref(t.data(), DataEncoder::template value<R>(),
                               S(t.shape()))
    {
    }

    template <typename R, rank_t r>
    basic_host_tensor_ref<R, r, Dim> ranked_as() const
    {
        return basic_host_tensor_ref<R, r, Dim>(
            data<R>(), this->shape_.template as_ranked<r>());
    }

    void *data() const { return this->data_.get(); }

    void *data_end() const
    {
        return static_cast<char *>(this->data_.get()) + this->data_size();
    }
};

template <typename DataEncoder, typename Dim>
class basic_raw_tensor_view
    : public raw_tensor_mixin<DataEncoder, basic_raw_shape<Dim>, host_memory,
                              readonly>
{
    using value_type_t = typename DataEncoder::value_type;

    using S = basic_raw_shape<Dim>;
    using mixin = raw_tensor_mixin<DataEncoder, S, host_memory, readonly>;

    template <typename R>
    const R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != this->value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<const R *>(this->data_.get());
    }

  public:
    template <typename... D>
    explicit basic_raw_tensor_view(const void *data,
                                   const value_type_t value_type, D... d)
        : basic_raw_tensor_view(data, value_type, S(d...))
    {
    }

    explicit basic_raw_tensor_view(const void *data,
                                   const value_type_t value_type,
                                   const S &shape)
        : mixin(data, value_type, shape)
    {
    }

    explicit basic_raw_tensor_view(
        const basic_raw_tensor_own<DataEncoder, Dim> &t)
        : mixin(t.data(), t.value_type(), t.shape())
    {
    }

    explicit basic_raw_tensor_view(
        const basic_raw_tensor_ref<DataEncoder, Dim> &t)
        : mixin(t.data(), t.value_type(), t.shape())
    {
    }

    template <typename R, rank_t r>
    explicit basic_raw_tensor_view(const basic_host_tensor_view<R, r, Dim> &t)
        : mixin(t.data(), DataEncoder::template value<R>(), S(t.shape()))
    {
    }

    template <typename R, rank_t r>
    basic_host_tensor_view<R, r, Dim> ranked_as() const
    {
        return basic_host_tensor_view<R, r, Dim>(
            data<R>(), this->shape_.template as_ranked<r>());
    }

    const void *data() const { return this->data_.get(); }

    const void *data_end() const
    {
        return static_cast<const char *>(this->data_.get()) + this->data_size();
    }
};
}  // namespace internal
}  // namespace ttl
