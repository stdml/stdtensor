#pragma once
#include <gtest/gtest.h>  // TODO: don't depend on gtest

inline void make_unuse(void *) {}

#define UNUSED(e)                                                              \
    {                                                                          \
        make_unuse(&e);                                                        \
    }
