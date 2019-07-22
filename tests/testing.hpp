#pragma once
#include <algorithm>

#include <gtest/gtest.h>  // TODO: don't depend on gtest

inline void make_unuse(void *) {}

#define UNUSED(e)                                                              \
    {                                                                          \
        make_unuse(&e);                                                        \
    }

template <typename R, typename T> void test_data_end_raw(const T &t)
{
    ASSERT_EQ(static_cast<const char *>(t.data_end()) -
                  static_cast<const char *>(t.data()),
              static_cast<ptrdiff_t>(t.shape().size() * sizeof(R)));
}

template <typename F, typename T, size_t... I>
void do_apply(const F &f, const T &t, std::index_sequence<I...>)
{
    f(std::get<I>(t)...);
}

template <typename F, typename T> void apply(const F &f, const T &t)
{
    do_apply(f, t, std::make_index_sequence<std::tuple_size<T>::value>());
}

template <typename F, typename... I>
int for_all_permutations(const F &f, I... i)
{
    std::array<int, sizeof...(I)> a({static_cast<int>(i)...});
    std::sort(a.begin(), a.end());
    int p = 0;
    do {
        apply(f, a);  // std::apply since c++17
        ++p;
    } while (std::next_permutation(a.begin(), a.end()));
    return p;
}
