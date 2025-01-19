cmake_minimum_required(VERSION 3.15)
project(simulation)
set(PROJECT_NAME simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(HOST_FLAGS "-O3")
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(HOST_FLAGS "-g")
    endif()
    set(HOST_FLAGS          "${HOST_FLAGS} -Wall -Wextra -Wfatal-errors -fexceptions -fPIC")
elseif(MSVC)
    set(HOST_FLAGS "-O2")
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(HOST_FLAGS "-DEBUG")
    endif()
    set(HOST_FLAGS "${HOST_FLAGS} -W4 -utf-8 -EHa")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # TODO(ZhuFei): will export APIs only in the future
endif()
set(CUDA_FLAGS          "${HOST_FLAGS} -forward-unknown-to-host-compiler --extended-lambda --relocatable-device-code=true -Xcompiler")
set(CMAKE_C_FLAGS       "${HOST_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS     "${HOST_FLAGS} ${CMAKE_CXX_FLAGS}")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# add extern libraries
# find opengl
find_package(OpenGL REQUIRED)

# Enable cuda
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86 86)

cmake_policy(SET CMP0104 NEW)

# config asserts' path
set(ASSERTS_PATH ${CMAKE_SOURCE_DIR}/src/)

configure_file("include/asserts.h.in" ${CMAKE_SOURCE_DIR}/include/asserts.h)

# Add the source files
file(GLOB_RECURSE SOURCES 
        "src/*.cpp"
        "src/Common/*.cpp"
        "src/ImGui/*.cpp"
        "src/Renderer/*.cpp"
        "src/Renderer/Camera/*.cpp"
        "src/Renderer/Drawable/*.cpp"
        "src/Renderer/Manager/*.cpp"
        "src/Renderer/Postprocess/*.cpp"
        "src/Simulator/*.cpp"
        "src/Simulator/*.cu"
        "src/Simulator/*.cuh"
    )

# Add the include directories
include_directories(
        "include/"
        "src/"
        "src/Common"
        "src/ImGui"
        "src/Renderer"
        "src/Renderer/Camera"
        "src/Renderer/Drawable"
        "src/Renderer/Manager"
        "src/Renderer/Postprocess"
        "src/Simulator/"
    )

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_directories(${PROJECT_NAME} PRIVATE "lib")
target_include_directories(${PROJECT_NAME} PRIVATE "include")
target_link_libraries(${PROJECT_NAME} PRIVATE glew32.dll glew32.lib glew32s.lib glfw3.lib assimp-vc140-mt.dll assimp-vc140-mt.lib)
