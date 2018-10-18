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

LINK_DIRECTORIES(${PREFIX}/lib)

FUNCTION(ADD_GTEST target)
    ADD_EXECUTABLE(${target} ${ARGN} tests/test_all.cpp)
    TARGET_LINK_LIBRARIES(${target} gtest)
    ADD_DEPENDENCIES(${target} libgtest-dev)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
    ADD_TEST(NAME ${target} COMMAND ${target})
ENDFUNCTION()

ADD_GTEST(shape-test tests/shape_test.cpp)
ADD_GTEST(generic-shape-test tests/generic_shape_test.cpp)
ADD_GTEST(tensor-test tests/tensor_test.cpp)
ADD_GTEST(generic-tensor-test tests/generic_tensor_test.cpp)
