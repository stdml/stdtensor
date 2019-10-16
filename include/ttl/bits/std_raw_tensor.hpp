#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <ttl/bits/std_flat_tensor.hpp>
#include <ttl/bits/std_raw_shape.hpp>
#include <ttl/bits/std_tensor.hpp>

namespace ttl
{
namespace internal
{

template <typename DataEncoder, typename shape_t = basic_raw_shape<>>
class basic_raw_tensor;

template <typename DataEncoder, typename shape_t = basic_raw_shape<>>
class basic_raw_tensor_ref;

template <typename DataEncoder, typename shape_t = basic_raw_shape<>>
class basic_raw_tensor_view;

template <typename DataEncoder, typename shape_t> class basic_raw_tensor
{
    using value_type_t = typename DataEncoder::value_type;

    const value_type_t value_type_;
    const shape_t shape_;
    std::unique_ptr<char[]> data_;

  public:
    using encoder_type = DataEncoder;
    using shape_type = shape_t;

    template <typename... D>
    explicit basic_raw_tensor(const value_type_t value_type, D... d)
        : basic_raw_tensor(value_type, shape_t(d...))
    {
    }

    explicit basic_raw_tensor(const value_type_t value_type,
                              const shape_t &shape)
        : value_type_(value_type), shape_(shape),
          data_(new char[shape_.size() * DataEncoder::size(value_type)])
    {
    }

    value_type_t value_type() const { return value_type_; }

    shape_t shape() const { return shape_; }

    size_t data_size() const
    {
        return encoder_type::size(value_type_) * shape_.size();
    }

    template <typename R> R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<R *>(data_.get());
    }

    void *data() const { return data_.get(); }

    void *data_end() const
    {
        return static_cast<char *>(data_.get()) +
               shape().size() * DataEncoder::size(value_type_);
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_host_tensor_ref<R, r, shape_type> ref_as() const
    {
        return ranked_as<basic_host_tensor_ref<R, r, shape_type>>();
    }

    template <typename R, rank_t r, typename shape_type = basic_shape<r>>
    basic_host_tensor_view<R, r, shape_type> view_as() const
    {
        return ranked_as<basic_host_tensor_view<R, r, shape_type>>();
    }

    template <typename R> basic_flat_tensor_ref<R, shape_t> typed_as() const
    {
        return basic_flat_tensor_ref<R, shape_t>(data<R>(), shape_);
    }

  private:
    template <typename T> T ranked_as() const
    {
        return T(data<typename T::value_type>(),
                 shape_.template as_ranked<T::rank>());
    }
};

template <typename DataEncoder, typename shape_t> class basic_raw_tensor_ref
{
    using value_type_t = typename DataEncoder::value_type;

    const value_type_t value_type_;
    const shape_t shape_;

    void *const data_;

    template <typename R> R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<R *>(data_);
    }

  public:
    using encoder_type = DataEncoder;
    using shape_type = shape_t;

    template <typename... D>
    explicit basic_raw_tensor_ref(void *data, const value_type_t value_type,
                                  D... d)
        : basic_raw_tensor_ref(data, value_type, shape_t(d...))
    {
    }

    explicit basic_raw_tensor_ref(void *data, const value_type_t value_type,
                                  const shape_t &shape)
        : value_type_(value_type), shape_(shape), data_(data)
    {
    }

    explicit basic_raw_tensor_ref(
        const basic_raw_tensor<DataEncoder, shape_t> &t)
        : value_type_(t.value_type()), shape_(t.shape()), data_(t.data())
    {
    }

    template <typename R, rank_t r, typename S>
    explicit basic_raw_tensor_ref(const basic_host_tensor_ref<R, r, S> &t)
        : value_type_(DataEncoder::template value<R>()), shape_(t.shape()),
          data_(t.data())
    {
    }

    template <typename R, rank_t r>
    basic_host_tensor_ref<R, r> ranked_as() const
    {
        return basic_host_tensor_ref<R, r>(data<R>(),
                                           shape_.template as_ranked<r>());
    }

    value_type_t value_type() const { return value_type_; }

    shape_t shape() const { return shape_; }

    size_t data_size() const
    {
        return encoder_type::size(value_type_) * shape_.size();
    }

    void *data() const { return data_; }

    void *data_end() const
    {
        return static_cast<char *>(data_) +
               shape().size() * DataEncoder::size(value_type_);
    }
};

template <typename DataEncoder, typename shape_t> class basic_raw_tensor_view
{
    using value_type_t = typename DataEncoder::value_type;

    const value_type_t value_type_;
    const shape_t shape_;

    const void *const data_;

    template <typename R> const R *data() const
    {
        // TODO: use contracts of c++20
        if (DataEncoder::template value<R>() != value_type_) {
            throw std::invalid_argument("invalid scalar type");
        }
        return reinterpret_cast<const R *>(data_);
    }

  public:
    using encoder_type = DataEncoder;
    using shape_type = shape_t;

    template <typename... D>
    explicit basic_raw_tensor_view(const void *data,
                                   const value_type_t value_type, D... d)
        : basic_raw_tensor_view(data, value_type, shape_t(d...))
    {
    }

    explicit basic_raw_tensor_view(const void *data,
                                   const value_type_t value_type,
                                   const shape_t &shape)
        : value_type_(value_type), shape_(shape), data_(data)
    {
    }

    explicit basic_raw_tensor_view(
        const basic_raw_tensor<DataEncoder, shape_t> &t)
        : value_type_(t.value_type()), shape_(t.shape()), data_(t.data())
    {
    }

    explicit basic_raw_tensor_view(
        const basic_raw_tensor_ref<DataEncoder, shape_t> &t)
        : value_type_(t.value_type()), shape_(t.shape()), data_(t.data())
    {
    }

    template <typename R, rank_t r, typename S>
    explicit basic_raw_tensor_view(const basic_host_tensor_view<R, r, S> &t)
        : value_type_(DataEncoder::template value<R>()), shape_(t.shape()),
          data_(t.data())
    {
    }

    template <typename R, rank_t r>
    basic_host_tensor_view<R, r> ranked_as() const
    {
        return basic_host_tensor_view<R, r>(data<R>(),
                                            shape_.template as_ranked<r>());
    }

    value_type_t value_type() const { return value_type_; }

    shape_t shape() const { return shape_; }

    size_t data_size() const
    {
        return encoder_type::size(value_type_) * shape_.size();
    }

    const void *data() const { return data_; }

    const void *data_end() const
    {
        return static_cast<const char *>(data_) +
               shape().size() * DataEncoder::size(value_type_);
    }
};
}  // namespace internal
}  // namespace ttl
