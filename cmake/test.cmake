INCLUDE(ExternalProject)
EXTERNALPROJECT_ADD(libgtest-dev
                    GIT_REPOSITORY
                    https://github.com/google/googletest
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/3rdparty)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/include)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/lib)

ADD_EXECUTABLE(all_tests
               tests/shape_test.cpp
               tests/tensor_test.cpp
               tests/test_all.cpp)

TARGET_LINK_LIBRARIES(all_tests gtest)
ADD_DEPENDENCIES(all_tests libgtest-dev)

ADD_TEST(NAME all_tests COMMAND all_tests)
