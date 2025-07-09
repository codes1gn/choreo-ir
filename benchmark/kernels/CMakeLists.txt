# CUDA Kernel Implementations CMakeLists.txt

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Collect kernel source files
file(GLOB_RECURSE KERNEL_SOURCES
    "*.cpp"
    "*.cu"
)

# Create kernel implementations library
if(KERNEL_SOURCES)
    add_library(choreo-ir-kernels ${KERNEL_SOURCES})
    
    # Set CUDA properties
    set_property(TARGET choreo-ir-kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET choreo-ir-kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Link CUDA libraries
    target_link_libraries(choreo-ir-kernels
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
    )
    
    # Set output directory
    set_target_properties(choreo-ir-kernels PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Create individual kernel test executables
function(add_kernel_test name source)
    add_executable(${name} ${source})
    
    target_link_libraries(${name}
        choreo-ir-kernels
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
    
    set_property(TARGET ${name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

# Add kernel test targets if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_kernels.cpp)
    add_kernel_test(test_matmul_kernels test_matmul_kernels.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_conv_kernels.cpp)
    add_kernel_test(test_conv_kernels test_conv_kernels.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_elementwise_kernels.cpp)
    add_kernel_test(test_elementwise_kernels test_elementwise_kernels.cpp)
endif() 