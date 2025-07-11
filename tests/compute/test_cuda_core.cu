/**
 * @file test_cuda_core.cpp
 * @brief Test suite for CUDA core micro-kernel implementation
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <random>
#include <iostream>

#include "choreo-ir/compute/cuda_core.hpp"
#include "choreo-ir/tensor/tensor.hpp"
#include "choreo-ir/core/types.hpp"

using namespace choreo_ir;
using namespace choreo_ir::compute;

class CudaCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
    
    void TearDown() override {
        cudaDeviceReset();
    }
    
    // Helper function to create random tensor
    template<typename T>
    Tensor<T> create_random_tensor(const Shape& shape, MemoryType memory_type = MemoryType::DEVICE) {
        Tensor<T> tensor(shape, memory_type);
        
        // Generate random data on host
        std::vector<T> host_data(shape.numel());
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < host_data.size(); ++i) {
                host_data[i] = dis(gen);
            }
        } else if constexpr (std::is_same_v<T, __half>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < host_data.size(); ++i) {
                host_data[i] = __float2half(dis(gen));
            }
        }
        
        // Copy to device if needed
        if (memory_type == MemoryType::DEVICE) {
            tensor.copy_from_host(host_data.data());
        } else {
            std::copy(host_data.begin(), host_data.end(), tensor.data());
        }
        
        return tensor;
    }
    
    // Helper function to compare tensors
    template<typename T>
    bool compare_tensors(const Tensor<T>& A, const Tensor<T>& B, T tolerance = T(1e-5)) {
        if (A.shape() != B.shape()) return false;
        
        std::vector<T> host_A(A.shape().numel());
        std::vector<T> host_B(B.shape().numel());
        
        A.copy_to_host(host_A.data());
        B.copy_to_host(host_B.data());
        
        for (size_t i = 0; i < host_A.size(); ++i) {
            if (std::abs(host_A[i] - host_B[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// Test basic functionality
TEST_F(CudaCoreTest, BasicMatmul) {
    // Create test matrices
    Shape shape_A({32, 32});
    Shape shape_B({32, 32});
    Shape shape_C({32, 32});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    auto C = create_random_tensor<float>(shape_C);
    
    // Perform matrix multiplication
    CudaCoreConfig config;
    cuda_core_matmul(A, B, C, config);
    
    // Verify result is not all zeros
    std::vector<float> host_C(shape_C.numel());
    C.copy_to_host(host_C.data());
    
    bool has_non_zero = false;
    for (float val : host_C) {
        if (val != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_non_zero) << "Matrix multiplication result should not be all zeros";
}

// Test GEMM with alpha and beta
TEST_F(CudaCoreTest, GEMMWithScaling) {
    Shape shape_A({16, 16});
    Shape shape_B({16, 16});
    Shape shape_C({16, 16});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    auto C = create_random_tensor<float>(shape_C);
    
    float alpha = 2.0f;
    float beta = 0.5f;
    
    // Store original C for comparison
    auto C_original = C;
    
    // Perform GEMM
    CudaCoreConfig config;
    cuda_core_gemm(A, B, C, alpha, beta, config);
    
    // Verify result is different from original
    EXPECT_FALSE(compare_tensors(C, C_original, 1e-6f));
}

// Test half precision
TEST_F(CudaCoreTest, HalfPrecision) {
    Shape shape_A({16, 16});
    Shape shape_B({16, 16});
    Shape shape_C({16, 16});
    
    auto A = create_random_tensor<__half>(shape_A);
    auto B = create_random_tensor<__half>(shape_B);
    auto C = create_random_tensor<__half>(shape_C);
    
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    
    // Perform GEMM
    CudaCoreConfig config;
    cuda_core_gemm(A, B, C, alpha, beta, config);
    
    // Verify result is not all zeros
    std::vector<__half> host_C(shape_C.numel());
    C.copy_to_host(host_C.data());
    
    bool has_non_zero = false;
    for (__half val : host_C) {
        if (__half2float(val) != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_non_zero) << "Half precision matrix multiplication result should not be all zeros";
}

// Test different matrix sizes
TEST_F(CudaCoreTest, DifferentSizes) {
    std::vector<std::pair<Shape, Shape>> test_cases = {
        {{8, 8}, {8, 8}},
        {{16, 16}, {16, 16}},
        {{32, 32}, {32, 32}},
        {{64, 64}, {64, 64}},
        {{16, 32}, {32, 16}},
        {{32, 16}, {16, 32}}
    };
    
    for (const auto& [shape_A, shape_B] : test_cases) {
        Shape shape_C({shape_A[0], shape_B[1]});
        
        auto A = create_random_tensor<float>(shape_A);
        auto B = create_random_tensor<float>(shape_B);
        auto C = create_random_tensor<float>(shape_C);
        
        // Perform matrix multiplication
        CudaCoreConfig config;
        cuda_core_matmul(A, B, C, config);
        
        // Verify result dimensions
        EXPECT_EQ(C.shape()[0], shape_A[0]);
        EXPECT_EQ(C.shape()[1], shape_B[1]);
    }
}

// Test configuration optimization
TEST_F(CudaCoreTest, ConfigurationOptimization) {
    Shape shape_A({64, 64});
    Shape shape_B({64, 64});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    
    // Get optimal configuration
    auto config = get_optimal_cuda_core_config(A, B);
    
    // Verify configuration is reasonable
    EXPECT_GT(config.block_m, 0);
    EXPECT_GT(config.block_n, 0);
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_GT(config.tile_k, 0);
}

// Test capability detection
TEST_F(CudaCoreTest, CapabilityDetection) {
    Shape shape_A({16, 16});
    Shape shape_B({16, 16});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    
    // Test CUDA core capability
    EXPECT_TRUE(can_use_cuda_core(A, B));
    
    // Test with CPU tensors (should fail)
    auto A_cpu = create_random_tensor<float>(shape_A, MemoryType::HOST);
    auto B_cpu = create_random_tensor<float>(shape_B, MemoryType::HOST);
    
    EXPECT_FALSE(can_use_cuda_core(A_cpu, B_cpu));
}

// Test error handling
TEST_F(CudaCoreTest, ErrorHandling) {
    Shape shape_A({16, 16});
    Shape shape_B({16, 16});
    Shape shape_C({16, 16});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    auto C = create_random_tensor<float>(shape_C);
    
    // Test with invalid configuration
    CudaCoreConfig invalid_config;
    invalid_config.block_m = 0;
    invalid_config.block_n = 0;
    
    EXPECT_THROW(validate_cuda_core_contract(A, B, C, invalid_config), 
                 std::runtime_error);
}

// Performance test
TEST_F(CudaCoreTest, PerformanceTest) {
    Shape shape_A({512, 512});
    Shape shape_B({512, 512});
    Shape shape_C({512, 512});
    
    auto A = create_random_tensor<float>(shape_A);
    auto B = create_random_tensor<float>(shape_B);
    auto C = create_random_tensor<float>(shape_C);
    
    // Warm up
    CudaCoreConfig config;
    cuda_core_matmul(A, B, C, config);
    cudaDeviceSynchronize();
    
    // Performance measurement
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        cuda_core_matmul(A, B, C, config);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    
    std::cout << "CUDA Core GEMM Performance: " << avg_time_ms << " ms per iteration" << std::endl;
    
    // Verify performance is reasonable (should be < 100ms for 512x512)
    EXPECT_LT(avg_time_ms, 100.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 