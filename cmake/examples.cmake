FUNCTION(ADD_CPP_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
ENDFUNCTION()

FUNCTION(ADD_C_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    TARGET_LINK_LIBRARIES(${target} stdtensor)
ENDFUNCTION()

ADD_CPP_EXAMPLE(example-1 examples/example_1.cpp)
ADD_C_EXAMPLE(example-c-api examples/example_c_api.c)

OPTION(USE_OPENCV "Build examples with libopencv-dev" OFF)

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV)
    ADD_CPP_EXAMPLE(example-opencv examples/example_opencv.cpp)
    TARGET_LINK_LIBRARIES(example-opencv opencv_imgcodecs)
ENDIF()
