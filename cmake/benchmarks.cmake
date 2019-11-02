INCLUDE(${CMAKE_SOURCE_DIR}/cmake/gbench.cmake)

ADD_CUSTOM_TARGET(benchmarks)

FIND_PACKAGE(Threads REQUIRED)
FUNCTION(ADD_BENCH target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_USE_GBENCH(${target})
    IF(HAVE_CUDA)
        TARGET_LINK_LIBRARIES(${target} cudart)
    ENDIF()
    ADD_DEPENDENCIES(benchmarks ${target})
ENDFUNCTION()

ADD_BENCH(bench-1 tests/bench_cuda_tensor.cpp)
