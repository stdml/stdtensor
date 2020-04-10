#include "testing.hpp"

#include <ttl/experimental/raw_tensor>
#include <ttl/tensor>

TEST(raw_tensor_test, test1)
{
    using ttl::experimental::raw_tensor;

    using encoder = raw_tensor::encoder_type;
    using flat_shape = raw_tensor::shape_type;

    {
        raw_tensor t(encoder::value<float>());
        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(1));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        auto x = t.typed<float, 0>();
        static_assert(
            std::is_same<decltype(x), ttl::tensor_ref<float, 0>>::value, "");
    }
    {
        raw_tensor t(encoder::value<float>(), 1);
        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(1));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        auto x = t.typed<float, 1>();
        static_assert(
            std::is_same<decltype(x), ttl::tensor_ref<float, 1>>::value, "");
    }
    {
        raw_tensor t(encoder::value<float>(), 1, 2);
        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(2));
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.data<float>(), t.data());
        auto x = t.typed<float, 2>();
        static_assert(
            std::is_same<decltype(x), ttl::tensor_ref<float, 2>>::value, "");
    }
    {
        raw_tensor t(encoder::value<float>(), 1, 2, 3);
        ASSERT_EQ(t.value_type(), encoder::value<float>());
        ASSERT_EQ(t.shape().size(), static_cast<flat_shape::dimension_type>(6));
        ASSERT_EQ(t.data<float>(), t.data());
        auto x = t.typed<float, 3>();
        static_assert(
            std::is_same<decltype(x), ttl::tensor_ref<float, 3>>::value, "");
    }
}

TEST(raw_tensor_test, test_convert)
{
    using ttl::experimental::raw_ref;
    using ttl::experimental::raw_view;

    using ttl::experimental::raw_tensor;
    using encoder = raw_tensor::encoder_type;

    using ttl::experimental::raw_tensor_ref;
    using ttl::experimental::raw_tensor_view;

    using R = float;
    ttl::tensor<R, 4> t(10, 224, 244, 3);
    {
        raw_tensor_ref r(ref(t));
        raw_tensor_view v(view(t));

        raw_tensor_ref r1 = raw_ref(t);
        raw_tensor_view v1 = raw_view(t);

        GNU_UNUSED ttl::tensor_ref<R, 4> _tr = r.typed<R, 4>();
        GNU_UNUSED ttl::tensor_view<R, 4> _tv = v.typed<R, 4>();
    }
    {
        ttl::tensor_ref<float, 4> rt = ref(t);

        raw_tensor_ref r(rt);
        raw_tensor_view v(view(rt));

        raw_tensor_ref r1 = raw_ref(rt);
        raw_tensor_view v1 = raw_view(rt);
    }
    {
        ttl::tensor_view<float, 4> vt = view(t);

        raw_tensor_view v(vt);
        raw_tensor_view v1 = raw_view(vt);
    }

    {
        raw_tensor t(encoder::value<R>(), 3, 4, 5, 6);
        raw_tensor_ref r(t);
        raw_tensor_view v(t);
        raw_tensor_view v1(r);
    }
}

template <typename R, typename T>
void test_raw_accessors(const T &t)
{
    t.shape();
    t.value_type();
    test_data_end_raw<R>(t);

    t.data_size();
}

TEST(raw_tensor_test, test_data)
{
    using ttl::experimental::raw_tensor;
    using encoder = raw_tensor::encoder_type;

    using ttl::experimental::raw_ref;
    using ttl::experimental::raw_view;

    using ttl::experimental::raw_tensor_ref;
    using ttl::experimental::raw_tensor_view;

    using R = float;
    ttl::tensor<R, 4> t(10, 224, 244, 3);

    raw_tensor rt(encoder::value<R>(), 10, 224, 244, 3);
    raw_tensor_ref rr = raw_ref(t);
    raw_tensor_view rv = raw_view(t);

    test_raw_accessors<R>(rt);
    test_raw_accessors<R>(rr);
    test_raw_accessors<R>(rv);
}

TEST(raw_tensor_test, test_type_reification)
{
    using ttl::experimental::raw_tensor;
    raw_tensor t(raw_tensor::type<int>(), 2, 3);
    {
        bool caught = false;
        try {
            t.typed<float>();
        } catch (std::invalid_argument &e) {
            caught = true;
        }
        ASSERT_TRUE(caught);
    }
}

#include <ttl/experimental/flat_tensor>

TEST(raw_tensor_test, test_convert_to_flat)
{
    using ttl::experimental::raw_tensor;
    using ttl::experimental::raw_tensor_ref;
    using ttl::experimental::raw_tensor_view;
    using encoder = raw_tensor::encoder_type;
    using flat_shape = raw_tensor::shape_type;

    raw_tensor rt(encoder::value<float>(), 1, 2, 3);
    {
        static_assert(
            std::is_same<decltype(rt.typed<float>()),
                         ttl::experimental::flat_tensor_ref<float>>::value,
            "");
        ttl::experimental::flat_tensor_ref<float> ft = rt.typed<float>();
        ASSERT_EQ(ft.size(), static_cast<flat_shape::dimension_type>(6));
    }
    {
        const raw_tensor_ref rtr(rt);
        static_assert(
            std::is_same<decltype(rtr.typed<float>()),
                         ttl::experimental::flat_tensor_ref<float>>::value,
            "");
    }
    {
        const raw_tensor_view rtv(rt);
        static_assert(
            std::is_same<decltype(rtv.typed<float>()),
                         ttl::experimental::flat_tensor_view<float>>::value,
            "");
    }
}
