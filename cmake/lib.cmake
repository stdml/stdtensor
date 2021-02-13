# TODO: deprecate
ADD_LIBRARY(stdtensor src/tensor.cpp)
INSTALL(TARGETS stdtensor ARCHIVE DESTINATION lib)

# libtensor will be shared by python/haskell bindings
ADD_LIBRARY(tensor src/tensor2.cpp)
INSTALL(TARGETS tensor ARCHIVE DESTINATION lib)
SET_PROPERTY(TARGET tensor PROPERTY CXX_STANDARD 17)
