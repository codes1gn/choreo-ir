# Performance benchmark suite

# Find required packages
find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Create performance benchmark executable
add_executable(benchmark_suite
    benchmark_suite.cpp
)

target_link_libraries(benchmark_suite
    PRIVATE
    choreo-ir-core
    choreo-ir-baselines
    benchmark::benchmark
    benchmark::benchmark_main
    CUDA::cublas
    CUDA::cudart
    CUDA::curand
)

target_include_directories(benchmark_suite
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/benchmark/baselines
)

# 删除所有 target_compile_features 相关行，只保留 set_property 方式
set_property(TARGET benchmark_suite PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_suite PROPERTY CXX_STANDARD_REQUIRED ON)

# Add CUDA support
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # 注释掉 CUDA_STANDARD 14，避免全局语言标准要求
    # set_property(TARGET benchmark_suite PROPERTY CUDA_STANDARD 14)
    # set_property(TARGET benchmark_suite PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET benchmark_suite PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add compiler-specific flags
target_compile_options(benchmark_suite PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)

# Add definitions for optional libraries
target_compile_definitions(benchmark_suite PRIVATE
    $<$<TARGET_EXISTS:CUDA::cudnn>:HAVE_CUDNN>
    $<$<TARGET_EXISTS:CUDA::curand>:HAVE_CURAND>
)

# Create custom targets for running different benchmark categories
add_custom_target(run_matmul_benchmarks
    COMMAND benchmark_suite --benchmark_filter=".*Matmul.*" 
    --benchmark_format=json 
    --benchmark_out=matmul_benchmark_results.json
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running matrix multiplication benchmarks"
)

add_custom_target(run_conv_benchmarks
    COMMAND benchmark_suite --benchmark_filter=".*Conv.*" 
    --benchmark_format=json 
    --benchmark_out=conv_benchmark_results.json
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running convolution benchmarks"
)

add_custom_target(run_tensor_core_benchmarks
    COMMAND benchmark_suite --benchmark_filter=".*TensorCore.*" 
    --benchmark_format=json 
    --benchmark_out=tensor_core_benchmark_results.json
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tensor core benchmarks"
)

add_custom_target(run_batch_benchmarks
    COMMAND benchmark_suite --benchmark_filter=".*Batch.*" 
    --benchmark_format=json 
    --benchmark_out=batch_benchmark_results.json
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running batch operation benchmarks"
)

add_custom_target(run_memory_benchmarks
    COMMAND benchmark_suite --benchmark_filter=".*Memory.*" 
    --benchmark_format=json 
    --benchmark_out=memory_benchmark_results.json
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running memory bandwidth benchmarks"
)

# Master benchmark target
add_custom_target(run_all_performance_benchmarks
    COMMAND benchmark_suite 
    --benchmark_format=json 
    --benchmark_out=complete_benchmark_results.json
    --benchmark_repetitions=3
    --benchmark_report_aggregates_only=true
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running complete performance benchmark suite"
)

# Performance regression testing target
add_custom_target(run_regression_tests
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_regression_tests.py 
    --build-dir ${CMAKE_BINARY_DIR}
    --iterations 5
    --plots
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running performance regression tests"
)

# Create baseline target
add_custom_target(create_performance_baseline
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_regression_tests.py 
    --build-dir ${CMAKE_BINARY_DIR}
    --iterations 10
    --save-baseline
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Creating performance baseline"
)

# Quick performance check (for CI)
add_custom_target(quick_performance_check
    COMMAND benchmark_suite 
    --benchmark_filter=".*small.*|.*medium.*" 
    --benchmark_format=json 
    --benchmark_out=quick_benchmark_results.json
    --benchmark_min_time=0.1
    DEPENDS benchmark_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running quick performance check"
)

# Install benchmark executable
install(TARGETS benchmark_suite
    RUNTIME DESTINATION bin
    COMPONENT benchmarks
)

# Install Python regression script
install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/run_regression_tests.py
    DESTINATION bin
    COMPONENT benchmarks
)

# Create benchmark configuration summary
get_target_property(BENCHMARK_COMPILE_FEATURES benchmark_suite COMPILE_FEATURES)
get_target_property(BENCHMARK_LINK_LIBRARIES benchmark_suite LINK_LIBRARIES)

message(STATUS "Performance benchmark configuration:")
message(STATUS "  Executable: benchmark_suite")
message(STATUS "  C++ Standard: C++14")
message(STATUS "  CUDA Support: ${CMAKE_CUDA_COMPILER}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "  CUDA Standard: CUDA 14")
endif()
message(STATUS "  Optimization: -O3")
message(STATUS "  Linked Libraries: ${BENCHMARK_LINK_LIBRARIES}")

# Add test that verifies benchmarks can run
add_test(NAME BenchmarkSmokeTest
    COMMAND benchmark_suite 
    --benchmark_filter=".*small_square.*" 
    --benchmark_min_time=0.01
    --benchmark_repetitions=1
)

set_tests_properties(BenchmarkSmokeTest PROPERTIES
    TIMEOUT 60
    LABELS "benchmark;smoke"
) 