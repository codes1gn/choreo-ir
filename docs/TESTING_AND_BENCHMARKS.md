# Testing and Benchmark Framework

This document describes the comprehensive testing and benchmarking infrastructure for Choreo-IR's ideal programming experience.

## Overview

We've built a complete testing and performance framework that validates our ideal API works correctly and performs competitively against established baselines like cuBLAS and cuDNN.

## Framework Components

### 1. Unit Tests (`tests/`)

**File**: `tests/test_end2end.cpp`

Comprehensive unit tests for the ideal programming experience API:

- **Tensor Creation & Properties**: Test different ways to create tensors (zeros, ones, random)
- **Core Programming Pattern**: Validate `C = A * B` syntax works correctly
- **Mixed Precision**: Test automatic tensor core usage with half precision inputs
- **Various Matrix Sizes**: Ensure robustness across different problem sizes
- **Batch Operations**: Test natural batch multiplication syntax
- **Convolution**: Basic 2D convolution testing
- **Memory Management**: Verify automatic cleanup and memory handling
- **Error Handling**: Test edge cases and error conditions
- **Performance Regression**: Built-in performance checks to catch regressions

**Key Features**:
- Compares results against cuBLAS for accuracy validation
- Measures performance to ensure competitive execution times
- Tests both host and device tensor operations
- Validates tensor core usage through performance metrics

### 2. Performance Benchmarks (`benchmark/performance/`)

**File**: `benchmark/performance/benchmark_suite.cpp`

Comprehensive benchmark suite comparing our ideal API against industry baselines:

#### Benchmark Categories

1. **Matrix Multiplication Benchmarks**
   - Float vs Half precision
   - Our API vs cuBLAS baseline
   - Multiple matrix sizes (small to extra-large)
   - Tensor core vs regular computation

2. **Mixed Precision Benchmarks**
   - Half precision input, float output
   - Automatic tensor core engagement
   - Performance comparison with cuBLAS GemmEx

3. **Batch Operations**
   - Batch matrix multiplication
   - Different batch sizes and matrix dimensions
   - Strided batch operations

4. **Convolution Benchmarks**
   - 2D convolution operations
   - Various input sizes (ResNet-like workloads)
   - Integration with cuDNN baselines

5. **Memory Bandwidth Tests**
   - Tensor copy operations
   - Memory transfer efficiency
   - Different data sizes

#### Performance Metrics

- **TFLOPS**: Computational throughput
- **Memory Bandwidth**: GB/s for memory-bound operations
- **Latency**: Operation completion time
- **Tensor Core Utilization**: Automatic detection and usage

### 3. Performance Regression Testing (`scripts/`)

**File**: `scripts/run_regression_tests.py`

Automated performance regression detection system:

#### Features

- **Baseline Management**: Save/load performance baselines
- **Automated Comparison**: Compare current performance against historical data
- **Regression Detection**: Configurable thresholds for performance changes
- **HTML Reports**: Detailed performance analysis reports
- **Visualization**: Performance trend graphs and comparisons
- **CI/CD Integration**: Fail builds on significant performance regressions

#### Usage

```bash
# Create initial baseline
python scripts/run_regression_tests.py --save-baseline

# Run regression check
python scripts/run_regression_tests.py --plots

# CI/CD integration (fail on regression)
python scripts/run_regression_tests.py --fail-on-regression
```

### 4. Baseline Implementations (`benchmark/baselines/`)

**File**: `benchmark/baselines/cublas_impl.hpp`

Wrapper implementations for established libraries:

- **cuBLAS**: Matrix operations, GEMM variants, batch operations
- **cuDNN**: Convolution operations, activation functions
- **Helper Classes**: Memory management, timing utilities

## Build Integration

### CMake Configuration

- **Unit Tests**: Integrated with CTest framework
- **Benchmarks**: Google Benchmark integration
- **CUDA Support**: Automatic architecture detection
- **Optional Features**: Memory checking (Valgrind), coverage analysis
- **Sanitizers**: AddressSanitizer, UBSan for debugging

### Build Targets

```bash
# Run all tests
make run_all_tests

# Run specific test categories
make run_unit_tests
make run_integration_tests
make run_performance_tests

# Run benchmarks
make run_all_performance_benchmarks
make run_matmul_benchmarks
make run_tensor_core_benchmarks

# Performance regression
make run_regression_tests
make create_performance_baseline
```

## Quick Demo

**File**: `scripts/run_quick_demo.py`

Complete demonstration script showing the ideal API in action:

```bash
# Full demo (build + test + benchmark)
./scripts/run_quick_demo.py

# Skip build step
./scripts/run_quick_demo.py --skip-build

# Check CUDA environment only
./scripts/run_quick_demo.py --cuda-check
```

## Ideal Programming Experience Examples

### Basic Matrix Multiplication

```cpp
auto A = host_tensor<float>::random({1024, 512});
auto B = host_tensor<float>::random({512, 256});
auto C = A * B;  // Just like mathematical notation!
```

### Automatic Tensor Core Usage

```cpp
auto A = host_tensor<__half>::random({2048, 1024});
auto B = host_tensor<__half>::random({1024, 512});
auto C = host_tensor<float>::zeros({2048, 512});

matmul(A, B, C);  // Automatically uses tensor cores!
```

### Batch Operations

```cpp
auto batch_A = host_tensor<__half>::random({32, 256, 256});
auto batch_B = host_tensor<__half>::random({32, 256, 256});
auto batch_C = batch_matmul(batch_A, batch_B);
```

### Natural Convolution

```cpp
auto input = host_tensor<__half>::random({8, 64, 224, 224});
auto weight = host_tensor<__half>::random({128, 64, 3, 3});
auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
```

## Performance Validation

Our testing framework validates that the ideal API:

1. **Matches cuBLAS Accuracy**: Results within 1e-4 tolerance
2. **Achieves Competitive Performance**: ≥70% of cuBLAS TFLOPS
3. **Utilizes Tensor Cores**: Automatic engagement for supported operations
4. **Scales Efficiently**: Performance across various problem sizes
5. **Manages Memory Properly**: No leaks, efficient transfers

## Integration with Development Workflow

### Continuous Integration

- **Smoke Tests**: Quick validation that basic functionality works
- **Performance Gates**: Prevent merging code that degrades performance
- **Regression Tracking**: Historical performance monitoring
- **Multi-GPU Testing**: Validation across different hardware configurations

### Development Testing

- **Unit Test Coverage**: Comprehensive API coverage
- **Performance Profiling**: TFLOPS tracking for optimization work
- **Memory Analysis**: Leak detection and efficiency monitoring
- **Cross-Platform Testing**: Linux, Windows support validation

## Next Steps

1. **Expand Test Coverage**: Add more tensor operations and edge cases
2. **Real-World Workloads**: Integrate full neural network training/inference
3. **Multi-GPU Support**: Extend testing to multi-device scenarios
4. **Advanced Optimizations**: Custom kernel implementations and optimizations
5. **Python Bindings**: Extend testing to Python API when implemented

This framework ensures that Choreo-IR's ideal programming experience maintains both correctness and high performance as the library evolves. 