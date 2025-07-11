# Baselines module CMakeLists.txt

# Enable CUDA language
enable_language(CUDA)
# 注释掉全局 CUDA_STANDARD 17，避免语言标准要求
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Create baseline library
add_library(choreo-ir-baselines INTERFACE)

# Set include directories
target_include_directories(choreo-ir-baselines INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link CUDA libraries
target_link_libraries(choreo-ir-baselines INTERFACE 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Create baseline test executable
add_executable(test_baselines 
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_baselines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_baseline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/baseline_factory.cpp
)

# Set include directories for the test
target_include_directories(test_baselines PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDA_INCLUDE_DIRS}
)

# Set CUDA properties for the test
set_target_properties(test_baselines PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 80
)

# Set CUDA compile flags manually
set_target_properties(test_baselines PROPERTIES
    CUDA_NVCC_FLAGS "-std=c++14"
)

# Link libraries
target_link_libraries(test_baselines 
    choreo-ir-baselines
    gtest
    gtest_main
    pthread
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Add test
add_test(NAME BaselineTest COMMAND test_baselines)

# Force include of all baseline headers to ensure template instantiations
target_compile_definitions(test_baselines PRIVATE
    FORCE_INCLUDE_BASELINES=1
) 