FUNCTION(BUILD_STDML_TENSOR_LIB target)
    # libtensor will be shared by python/haskell bindings
    FILE(GLOB_RECURSE srcs ${CMAKE_SOURCE_DIR}/src/stdml/*)
    ADD_LIBRARY(${target} SHARED ${srcs})
    SET_TARGET_PROPERTIES(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    TARGET_LINK_LIBRARIES(${target} dl)
    SET_PROPERTY(TARGET ${target} PROPERTY CXX_STANDARD 17)
    INSTALL(TARGETS ${target} DESTINATION lib)
ENDFUNCTION()
