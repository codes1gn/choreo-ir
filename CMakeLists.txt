cmake_minimum_required(VERSION 3.18)
project(choreo-ir LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (compute capability >= 7.0)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(benchmark)
add_subdirectory(examples)

# Create header-only library target
add_library(choreo-ir INTERFACE)
target_include_directories(choreo-ir INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
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