FUNCTION(ADD_CPP_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
ENDFUNCTION()

FUNCTION(ADD_C_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_LINK_LIBRARIES(${target} tensor)
ENDFUNCTION()

ADD_CPP_EXAMPLE(example-1 examples/example_1.cpp)
ADD_C_EXAMPLE(example-c examples/example_c.c)

OPTION(USE_OPENCV "Build examples with libopencv-dev" OFF)

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV)
    ADD_CPP_EXAMPLE(example-opencv examples/example_opencv.cpp)
    TARGET_LINK_LIBRARIES(example-opencv opencv_imgcodecs)
ENDIF()

FILE(GLOB examples examples/simple/*.cpp)
FOREACH(src ${examples})
    GET_FILENAME_COMPONENT(name ${src} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_EXECUTABLE(${name} ${src})
ENDFOREACH()
