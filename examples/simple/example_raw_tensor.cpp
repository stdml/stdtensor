#include <ttl/experimental/flat_tensor>
#include <ttl/experimental/raw_tensor>

void f(const ttl::experimental::flat_tensor_view<int> &t) {}

void f(const ttl::experimental::flat_tensor_view<float> &t) {}

void f(const ttl::experimental::raw_tensor_view &t)
{
    using T = std::remove_reference<decltype(t)>::type;
    switch (t.value_type()) {
    case T::type<int>():
        f(t.typed<int>());
        break;
    case T::type<float>():
        f(t.typed<float>());
        break;
    }
}

int main()
{
    using ttl::experimental::raw_tensor;
    raw_tensor x(raw_tensor::type<int>(), 2, 3);
    raw_tensor y(raw_tensor::type<float>(), 2, 3);
    f(ttl::experimental::raw_tensor_view(x));
    f(ttl::experimental::raw_tensor_view(y));
    // f(ttl::view(x));
    // f(ttl::view(y));
    return 0;
}
