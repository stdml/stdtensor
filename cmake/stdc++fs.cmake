IF(APPLE)
    ADD_DEFINITIONS(-DHAVE_STD_CPP_FS)
ELSEIF(MSVC)
    ADD_DEFINITIONS(-DHAVE_STD_CPP_FS)
ELSE()
    LINK_LIBRARIES(stdc++fs)
ENDIF()
