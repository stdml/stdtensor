INCLUDE(${CMAKE_SOURCE_DIR}/cmake/gtest.cmake)

FUNCTION(ADD_UNIT_TEST target)
    ADD_EXECUTABLE(${target} ${ARGN} tests/main.cpp)
    TARGET_USE_GTEST(${target})
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${CMAKE_SOURCE_DIR}/tests/include)
    IF(BUILD_LIB)
        TARGET_LINK_LIBRARIES(${target} stdtensor)
    ENDIF()
    IF(HAVE_CUDA)
        TARGET_LINK_LIBRARIES(${target} cudart)
    ENDIF()
    ADD_TEST(NAME ${target} COMMAND ${target})
ENDFUNCTION()

FUNCTION(ADD_UNIT_TESTS)
    FOREACH(t ${ARGN})
        GET_FILENAME_COMPONENT(name ${t} NAME_WE)
        STRING(REPLACE "_" "-" name ${name})
        ADD_UNIT_TEST(${name} ${t})
    ENDFOREACH()
ENDFUNCTION()

FILE(GLOB tests tests/test_*.cpp)
IF(MSVC)
    LIST(REMOVE_ITEM tests ${CMAKE_SOURCE_DIR}/tests/test_loc.cpp)
    LIST(REMOVE_ITEM tests ${CMAKE_SOURCE_DIR}/tests/test_shape_debug.cpp)
    LIST(REMOVE_ITEM tests ${CMAKE_SOURCE_DIR}/tests/test_tensor_type.cpp)
ENDIF()
ADD_UNIT_TESTS(${tests})
