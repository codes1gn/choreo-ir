# Benchmark suite for Choreo-IR performance testing

# Find required packages
find_package(benchmark REQUIRED)
find_package(CUDA REQUIRED)

# Create baseline implementations library
add_subdirectory(baselines)

# Create performance benchmark suite
add_subdirectory(performance)

# Create kernel examples that demonstrate the ideal API usage
add_subdirectory(kernels)

# Create comparisons with other libraries
add_subdirectory(comparisons)

# Summary message
message(STATUS "Benchmark suite configured:")
message(STATUS "  Baseline implementations: benchmark/baselines/")
message(STATUS "  Performance benchmarks: benchmark/performance/")
message(STATUS "  Kernel examples: benchmark/kernels/")
message(STATUS "  Library comparisons: benchmark/comparisons/") 