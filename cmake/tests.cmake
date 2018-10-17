INCLUDE(ExternalProject)

SET(GTEST_GIT_URL
    https://github.com/google/googletest.git
    CACHE
    STRING
    "URL for clone gtest")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(libgtest-dev
                    GIT_REPOSITORY
                    ${GTEST_GIT_URL}
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DCMAKE_CXX_FLAGS=-std=c++11
                    -Dgtest_disable_pthreads=1
                    -DBUILD_GMOCK=0)

INCLUDE_DIRECTORIES(${PREFIX}/include)
LINK_DIRECTORIES(${PREFIX}/lib)

ADD_EXECUTABLE(all_tests
               tests/shape_test.cpp
               tests/tensor_test.cpp
               tests/test_all.cpp)

TARGET_LINK_LIBRARIES(all_tests gtest)
ADD_DEPENDENCIES(all_tests libgtest-dev)

ADD_TEST(NAME all_tests COMMAND all_tests)
