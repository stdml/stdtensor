#include "testing.hpp"

#include <ttl/consistent_variable>

TEST(consistent_variable_test, test_1)
{
    ttl::consistent_variable<int> i;
    i = 1;
    try {
        i = 2;
        ASSERT_TRUE(false);
    } catch (...) {
        ASSERT_TRUE(true);
    }

    try {
        i = 1;
        ASSERT_TRUE(true);
    } catch (...) {
        ASSERT_TRUE(false);
    }
}

TEST(consistent_variable_test, test_2)
{
    ttl::consistent_variable<int> i;
    try {
        int j = i;
        ASSERT_TRUE(false);
        ASSERT_EQ(j, 0);
    } catch (...) {
        ASSERT_TRUE(true);
    }

    i = 1;
    try {
        int j = i;
        ASSERT_TRUE(true);
        ASSERT_EQ(j, 1);
    } catch (...) {
        ASSERT_TRUE(false);
    }
}
