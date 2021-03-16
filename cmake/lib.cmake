# TODO: deprecate
ADD_LIBRARY(stdtensor src/tensor.cpp)
INSTALL(TARGETS stdtensor ARCHIVE DESTINATION lib)

# libtensor will be shared by python/haskell bindings
FILE(GLOB_RECURSE srcs ${CMAKE_SOURCE_DIR}/src/stdml/*)
ADD_LIBRARY(tensor ${srcs})
TARGET_LINK_LIBRARIES(tensor dl)

INSTALL(TARGETS tensor ARCHIVE DESTINATION lib)
SET_PROPERTY(TARGET tensor PROPERTY CXX_STANDARD 17)
