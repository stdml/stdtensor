OPTION(BUILD_GTEST "Build gtest from source." OFF)

IF(BUILD_GTEST)
    INCLUDE(ExternalProject)

    SET(GTEST_GIT_URL
        https://github.com/google/googletest.git
        CACHE STRING "URL for clone gtest")

    SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)
    LINK_DIRECTORIES(${PREFIX}/lib)
    LINK_DIRECTORIES(${PREFIX}/lib64)

    EXTERNALPROJECT_ADD(
        libgtest-dev-repo
        GIT_REPOSITORY ${GTEST_GIT_URL}
        PREFIX ${PREFIX}
        # https://stackoverflow.com/questions/19024259/how-to-change-the-build-
        # type-to-release-mode-in-cmake
        BUILD_COMMAND cmake --build . --config Release
        INSTALL_COMMAND cmake --install . --config Release
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX}
                   -DCMAKE_CXX_FLAGS=-std=c++11 -Dgtest_disable_pthreads=1
                   -DCMAKE_BUILD_TYPE=Release -DBUILD_GMOCK=0)

    LINK_DIRECTORIES(${PREFIX}/lib)

    FUNCTION(TARGET_USE_GTEST target)
        TARGET_INCLUDE_DIRECTORIES(${target} SYSTEM PRIVATE ${PREFIX}/include)
        TARGET_LINK_LIBRARIES(${target} gtest)
        ADD_DEPENDENCIES(${target} libgtest-dev-repo)
    ENDFUNCTION()
ELSE()
    FIND_PACKAGE(GTest REQUIRED)
    FIND_PACKAGE(Threads REQUIRED)
    FUNCTION(TARGET_USE_GTEST target)
        TARGET_INCLUDE_DIRECTORIES(${target} SYSTEM
                                   PRIVATE ${GTEST_INCLUDE_DIRS})
        TARGET_LINK_LIBRARIES(${target} ${GTEST_BOTH_LIBRARIES})
        TARGET_LINK_LIBRARIES(${target} Threads::Threads)
    ENDFUNCTION()
ENDIF()
