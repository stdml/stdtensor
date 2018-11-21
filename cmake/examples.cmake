ADD_EXECUTABLE(example-1 examples/example_1.cpp)
TARGET_INCLUDE_DIRECTORIES(example-1 PRIVATE ${CMAKE_SOURCE_DIR}/include)

OPTION(USE_OPENCV "Build examples with libopencv-dev" OFF)

IF(USE_OPENCV)
    FIND_PACKAGE(OpenCV)
    ADD_EXECUTABLE(example-opencv examples/example_opencv.cpp)
    TARGET_INCLUDE_DIRECTORIES(example-opencv
                               PRIVATE ${CMAKE_SOURCE_DIR}/include)
    TARGET_LINK_LIBRARIES(example-opencv opencv_imgcodecs)
ENDIF()
