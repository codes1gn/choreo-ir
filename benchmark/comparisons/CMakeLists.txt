# Benchmark Comparisons CMakeLists.txt

# Find Google Benchmark
find_package(benchmark REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Collect comparison source files
file(GLOB_RECURSE COMPARISON_SOURCES
    "*.cpp"
    "*.cu"
)

# Create benchmark comparison executable
if(COMPARISON_SOURCES)
    add_executable(choreo-ir-comparisons ${COMPARISON_SOURCES})
    
    # Set CUDA properties
    set_property(TARGET choreo-ir-comparisons PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET choreo-ir-comparisons PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Link required libraries
    target_link_libraries(choreo-ir-comparisons
        benchmark::benchmark
        benchmark::benchmark_main
        choreo-ir-kernels
        choreo-ir-baselines
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        pthread
    )
    
    # Link cuDNN if available
    if(CUDNN_FOUND)
        target_link_libraries(choreo-ir-comparisons ${CUDNN_LIBRARIES})
        target_include_directories(choreo-ir-comparisons PRIVATE ${CUDNN_INCLUDE_DIRS})
        target_compile_definitions(choreo-ir-comparisons PRIVATE HAVE_CUDNN)
    endif()
    
    # Set output directory
    set_target_properties(choreo-ir-comparisons PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Add custom target for running comparisons
    add_custom_target(run_comparisons
        COMMAND choreo-ir-comparisons --benchmark_format=json --benchmark_out=comparison_results.json
        DEPENDS choreo-ir-comparisons
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running benchmark comparisons"
    )
endif()

# Create individual comparison executables
function(add_comparison_benchmark name source)
    add_executable(${name} ${source})
    
    target_link_libraries(${name}
        benchmark::benchmark
        benchmark::benchmark_main
        choreo-ir-kernels
        choreo-ir-baselines
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        pthread
    )
    
    # Link cuDNN if available
    if(CUDNN_FOUND)
        target_link_libraries(${name} ${CUDNN_LIBRARIES})
        target_include_directories(${name} PRIVATE ${CUDNN_INCLUDE_DIRS})
        target_compile_definitions(${name} PRIVATE HAVE_CUDNN)
    endif()
    
    set_property(TARGET ${name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

# Add comparison benchmark targets if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/matmul_comparison.cpp)
    add_comparison_benchmark(matmul_comparison matmul_comparison.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/conv_comparison.cpp)
    add_comparison_benchmark(conv_comparison conv_comparison.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_comparison.cpp)
    add_comparison_benchmark(elementwise_comparison elementwise_comparison.cpp)
endif() 