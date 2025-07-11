cmake_minimum_required(VERSION 3.16)

# Try to read .cuda_arch file if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/.cuda_arch")
    file(READ "${CMAKE_SOURCE_DIR}/.cuda_arch" CUDA_ARCH_CONTENTS)
    string(STRIP "${CUDA_ARCH_CONTENTS}" CUDA_ARCH_CONTENTS)
    if(NOT "${CUDA_ARCH_CONTENTS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_CONTENTS}")
        message(STATUS "Read CUDA architectures from .cuda_arch: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Enable ccache for faster compilation
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "ccache found and enabled")
else()
    message(STATUS "ccache not found, compilation will not be cached")
endif()

# Set default CUDA architectures if not specified
# This must be done before the project() call
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Set common architectures for modern GPUs
    # 70: V100, 75: T4/RTX20xx, 80: A100, 86: RTX30xx, 89: RTX40xx
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Allow unsupported compiler for CUDA (needed for GCC 9.4 with CUDA 12.6)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(choreo-ir LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA using older method for CMake 3.16 compatibility
find_package(CUDA REQUIRED)

message(STATUS "Final CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add subdirectories - focus on tests for now
add_subdirectory(tests)
# TODO: Fix CUDAToolkit references in these directories
# add_subdirectory(benchmark)
# add_subdirectory(examples)

# Add baseline module
add_subdirectory(baselines)

# Create header-only library target
add_library(choreo-ir INTERFACE)
target_include_directories(choreo-ir INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set C++17 for the interface library
target_compile_features(choreo-ir INTERFACE cxx_std_17)

# Link CUDA libraries to the interface library using old method
target_link_libraries(choreo-ir INTERFACE 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Installation
install(DIRECTORY include/ DESTINATION include)
install(TARGETS choreo-ir EXPORT choreo-ir-targets)
install(EXPORT choreo-ir-targets DESTINATION lib/cmake/choreo-ir)

# Generate config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    choreo-ir-config-version.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES choreo-ir-config-version.cmake DESTINATION lib/cmake/choreo-ir) 