cmake_minimum_required(VERSION 3.16)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

function(add_ispcs ISPC_OBJECTS FILES)
    set(OBJECT_FILES "")
    foreach(SOURCE_FILE IN LISTS FILES)
        string(REPLACE ".ispc" ".ispc.o" OBJECT_FILE ${SOURCE_FILE})
        get_filename_component(OBJECT_FILE ${OBJECT_FILE} NAME)
        set(OBJECT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_FILE})
        set(OBJECT_FILES "${OBJECT_FILES};${OBJECT_FILE}")
        string(REPLACE ".ispc" ".h" HEADER_FILE ${SOURCE_FILE})
        add_custom_command(OUTPUT ${OBJECT_FILE}
            COMMAND ${ISPC_BIN} ${SOURCE_FILE} -o ${OBJECT_FILE} --target=${ISPC_TARGET} --arch=${ISPC_ARCH} --cpu=${ISPC_CPU} -h ${HEADER_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
    set(${ISPC_OBJECTS} ${OBJECT_FILES} PARENT_SCOPE)
endfunction(add_ispcs)

function (IsGnuCompiler)
    if((${CMAKE_C_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_C_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
        set(COMPILER_IS_GNU ON PARENT_SCOPE)
    else()
        set(COMPILER_IS_GNU OFF PARENT_SCOPE)
    endif()
endfunction()

set(PROJECT_NAME obb)
project(${PROJECT_NAME})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(AFTER ${PROJECT_ROOT})

#######################################################
# ISPC Compiler
#set(ISPC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/ispc)
#if(UNIX)
#    set(ISPC_BIN "${ISPC_ROOT}/bin/ispc")
#else()
#    set(ISPC_BIN "${ISPC_ROOT}/bin/ispc.exe")
#endif()
#set(ISPC_ARCH x86-64)
#set(ISPC_TARGET avx2)
#set(ISPC_CPU core-avx2)

########################################################################
# Sources
set(HEADERS "${PROJECT_ROOT}/OBB.h;${PROJECT_ROOT}/OBB_simd.h")
set(SOURCES "${PROJECT_ROOT}/OBB.cpp;${PROJECT_ROOT}/OBB_simd.cpp;${PROJECT_ROOT}/main.cpp")
#set(OBB_ISPCS "${PROJECT_ROOT}/OBB_ispc.ispc")

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${SOURCES})
#source_group("ispc" FILES ${OBB_ISPCS})

set(FILES ${HEADERS} ${SOURCES})

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

add_ispcs(OBB_ISPC_OBJECTS "${OBB_ISPCS}")
add_executable(${PROJECT_NAME} ${FILES} ${OBB_ISPC_OBJECTS})

IsGnuCompiler()

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MSBC /W4 /WX- /nologo /fp:precise /arch:AVX2 /Zc:wchar_t /TP /Gd")

    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++17")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /GL /GR- /DNDEBUG /EHsc-")

    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "/LTCG")

elseif(COMPILER_IS_GNU)
    if((${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64") OR (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86"))
        message("x86_64 or x86")
        set(DEFAULT_CXX_FLAGS "-Wall -Wextra -O2 -std=c++17 -std=gnu++17 -march=skylake")
    elseif((${CMAKE_SYSTEM_PROCESSOR} MATCHES "^armv7.*"))
        message("armv7")
        set(DEFAULT_CXX_FLAGS "-Wall -Wextra -O3 -std=c++17 -std=gnu++17 -march=armv7-a -mlittle-endian")
    elseif((${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64"))
        message("aarch64")
        set(DEFAULT_CXX_FLAGS "-Wall -Wextra -O3 -std=c++17 -std=gnu++17 -march=armv8-a -mlittle-endian")
    else()
        message("unsupported")
    endif()
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
elseif(APPLE)
endif()

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

