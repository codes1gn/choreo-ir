add_executable(test_wmma test_wmma.cu)
target_link_libraries(test_wmma gtest gtest_main pthread)
set_property(TARGET test_wmma PROPERTY CXX_STANDARD 17)
set_property(TARGET test_wmma PROPERTY CXX_STANDARD_REQUIRED ON)
set_target_properties(test_wmma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_wmma PROPERTIES CUDA_ARCHITECTURES 80)
# 强制 CUDA 文件用 c++14 编译，避免 CUDA17 错误
set_target_properties(test_wmma PROPERTIES CUDA_NVCC_FLAGS "-std=c++14")

add_executable(test_mma test_mma.cu)
target_link_libraries(test_mma gtest gtest_main pthread)
set_property(TARGET test_mma PROPERTY CXX_STANDARD 17)
set_property(TARGET test_mma PROPERTY CXX_STANDARD_REQUIRED ON)
set_target_properties(test_mma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_mma PROPERTIES CUDA_ARCHITECTURES 80)
# 强制 CUDA 文件用 c++14 编译，避免 CUDA17 错误
set_target_properties(test_mma PROPERTIES CUDA_NVCC_FLAGS "-std=c++14")

# Create compute module test executable
add_executable(test_cuda_core test_cuda_core.cu)

target_link_libraries(test_cuda_core
    PRIVATE
    choreo-ir-core
    gtest
    gtest_main
    pthread
    ${CUDA_LIBRARIES}
)

target_include_directories(test_cuda_core
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_cuda_core PROPERTY CXX_STANDARD 17)
set_property(TARGET test_cuda_core PROPERTY CXX_STANDARD_REQUIRED ON)

# 强制 CUDA 文件用 c++14 编译，避免 CUDA17 错误
set_target_properties(test_cuda_core PROPERTIES CUDA_NVCC_FLAGS "-std=c++14")
set_target_properties(test_cuda_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_cuda_core PROPERTIES CUDA_ARCHITECTURES 80)
