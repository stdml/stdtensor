#pragma once
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
