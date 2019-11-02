OPTION(BUILD_GBENCH "Build benchmark from source." OFF)

IF(BUILD_GBENCH)
    INCLUDE(ExternalProject)

    SET(GBENCH_GIT_URL
        https://github.com/google/benchmark.git
        CACHE STRING "URL for clone google benchmark")

    SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

    EXTERNALPROJECT_ADD(gbench-repo
                        GIT_REPOSITORY ${GBENCH_GIT_URL}
                        PREFIX ${PREFIX}
                        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX}
                                   -DCMAKE_BUILD_TYPE=Release
                                   -DCMAKE_CXX_FLAGS=-std=c++11
                                   -DBENCHMARK_ENABLE_TESTING=0)
    LINK_DIRECTORIES(${PREFIX}/lib)

    FUNCTION(TARGET_USE_GBENCH target)
        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
        TARGET_LINK_LIBRARIES(${target} benchmark)
        ADD_DEPENDENCIES(${target} gbench-repo)
    ENDFUNCTION()
ELSE()
    FIND_PACKAGE(benchmark REQUIRED)
    FUNCTION(TARGET_USE_GBENCH target)
        TARGET_LINK_LIBRARIES(${target} benchmark::benchmark)
    ENDFUNCTION()
ENDIF()
