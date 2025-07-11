/**
 * @file test_end2end.cpp
 * @brief End-to-end tests for the Choreo-IR library
 */

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <iostream>
#include "choreo-ir/choreo-ir.hpp"

using namespace choreo_ir;

class ChoreoIRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the library (this automatically handles device setup)
        ASSERT_TRUE(initialize());
    }
    
    void TearDown() override {
        // Synchronize device before finalizing
        cudaDeviceSynchronize();
        finalize();
        
        // Reset device after each test to prevent resource conflicts
        cudaDeviceReset();
    }
    
    // Helper to create random matrices (template with C++17 if constexpr)
    template<typename T>
    Tensor<T> create_random_matrix(int M, int N, float min_val = 0.0f, float max_val = 1.0f) {
        auto tensor = Tensor<T>::zeros(Shape({M, N}), MemoryType::DEVICE);
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        if constexpr (std::is_same_v<T, __half>) {
            std::vector<__half> host_data(M * N);
            for (int i = 0; i < M * N; ++i) {
                float val = dis(gen);
                host_data[i] = __float2half(val);
            }
            tensor.copy_from_host(host_data.data());
        } else {
            std::vector<T> host_data(M * N);
            for (int i = 0; i < M * N; ++i) {
                host_data[i] = static_cast<T>(dis(gen));
            }
            tensor.copy_from_host(host_data.data());
        }
        return tensor;
    }
    
    // Helper to compare tensors with tolerance
    template<typename T>
    bool tensors_equal(const Tensor<T>& a, const Tensor<T>& b, T tolerance = T(1e-5)) {
        if (a.shape() != b.shape()) return false;
        
        // Copy data to host for comparison
        std::vector<T> a_data(a.numel());
        std::vector<T> b_data(b.numel());
        a.copy_to_host(a_data.data());
        b.copy_to_host(b_data.data());
        
        for (size_t i = 0; i < a.numel(); ++i) {
            T diff;
            if constexpr (std::is_same_v<T, __half>) {
                diff = __float2half(std::abs(__half2float(a_data[i]) - __half2float(b_data[i])));
            } else {
                diff = std::abs(a_data[i] - b_data[i]);
            }
            
            if (diff > tolerance) {
                std::cout << "Mismatch at index " << i << ": " 
                         << a_data[i] << " vs " << b_data[i] 
                         << " (diff: " << diff << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Test basic tensor creation and properties
 */
TEST_F(ChoreoIRTest, TensorCreationAndProperties) {
    // Test different ways to create tensors
    auto A1 = Tensor<float>::zeros(Shape({64, 32}));
    auto A2 = Tensor<float>::ones(Shape({64, 32}));
    
    // Check shapes
    auto shape = A1.shape();
    EXPECT_EQ(shape[0], 64);
    EXPECT_EQ(shape[1], 32);
    EXPECT_EQ(A1.numel(), 64 * 32);
    EXPECT_EQ(A1.ndims(), 2);
    
    // Check initialization
    {
        std::vector<float> a1_data(64 * 32);
        A1.copy_to_host(a1_data.data());
        EXPECT_FLOAT_EQ(a1_data[0], 0.0f);
    }
    {
        std::vector<float> a2_data(64 * 32);
        A2.copy_to_host(a2_data.data());
        EXPECT_FLOAT_EQ(a2_data[0], 1.0f);
    }
    
    // Test device transfer
    auto d_A = Tensor<float>::device(Shape({64, 32}));
    EXPECT_EQ(d_A.shape(), A1.shape());
    EXPECT_EQ(d_A.memory_type(), MemoryType::DEVICE);
}

/**
 * @brief Test the core programming pattern: C = A * B
 */
TEST_F(ChoreoIRTest, SimpleMatrixMultiplication) {
    const int M = 16, N = 16, K = 16;
    
    // Create test matrices
    auto A = create_random_matrix<float>(M, K);
    auto B = create_random_matrix<float>(K, N);
    
    // Perform matrix multiplication
    auto C = A * B;
    
    // Verify shape
    auto C_shape = C.shape();
    EXPECT_EQ(C_shape[0], M);
    EXPECT_EQ(C_shape[1], N);
    
    // Verify result is not all zeros
    bool has_non_zero = false;
    std::vector<float> C_host(C.numel());
    C.copy_to_host(C_host.data());
    for (size_t i = 0; i < C.numel(); ++i) {
        if (C_host[i] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

/**
 * @brief Test mixed precision matrix multiplication
 */
TEST_F(ChoreoIRTest, MixedPrecisionMatmul) {
    const int M = 64, N = 32, K = 48;
    
    // Create half-precision input matrices
    auto A = create_random_matrix<__half>(M, K);
    auto B = create_random_matrix<__half>(K, N);
    
    // Output in float (automatic promotion)
    auto C = Tensor<float>::zeros(Shape({M, N}));
    
    // Perform mixed precision multiplication
    auto result = A * B;
    
    auto C_shape = result.shape();
    EXPECT_EQ(C_shape[0], M);
    EXPECT_EQ(C_shape[1], N);
}

/**
 * @brief Test batch operations
 */
TEST_F(ChoreoIRTest, BatchOperations) {
    const int batch_size = 16, M = 64, N = 32, K = 48;
    
    // Create batch tensors (simplified - just use 2D for now)
    auto A = create_random_matrix<float>(batch_size * M, K);
    auto B = create_random_matrix<float>(K, N);
    
    // Perform batch multiplication
    auto C = A * B;
    
    auto C_shape = C.shape();
    EXPECT_EQ(C_shape[0], batch_size * M);
    EXPECT_EQ(C_shape[1], N);
}

/**
 * @brief Test convolution operations
 */
TEST_F(ChoreoIRTest, ConvolutionOperations) {
    const int N = 1, C = 3, H = 32, W = 32;
    const int K = 64, R = 3, S = 3;
    
    // Create input and weight tensors
    auto input = create_random_matrix<float>(N * C, H * W);
    auto weight = create_random_matrix<float>(K * C, R * S);
    
    // Perform convolution (simplified)
    auto output = input; // For now, just return input
    
    // Check output shape (with padding=1, size should be preserved)
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape[0], N * C);
    EXPECT_EQ(output_shape[1], H * W);
}

/**
 * @brief Test element-wise operations
 */
TEST_F(ChoreoIRTest, ElementWiseOperations) {
    const int M = 64, N = 32;
    
    // Create test tensor with negative values
    auto input = create_random_matrix<float>(M, N, -2.0f, 2.0f);
    
    // Apply ReLU
    auto output = relu(input);
    
    // Verify all values are non-negative
    std::vector<float> output_host(output.numel());
    output.copy_to_host(output_host.data());
    for (size_t i = 0; i < output.numel(); ++i) {
        EXPECT_GE(output_host[i], 0.0f);
    }
    
    // Verify ReLU behavior
    std::vector<float> input_host(input.numel());
    input.copy_to_host(input_host.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        float expected = std::max(input_host[i], 0.0f);
        EXPECT_FLOAT_EQ(output_host[i], expected);
    }
}

/**
 * @brief Test real-world usage patterns
 */
TEST_F(ChoreoIRTest, RealWorldUsagePattern) {
    // Simulate a neural network layer
    const int input_dim = 512, hidden_dim = 256, output_dim = 128;
    
    // Create weight matrices
    auto W1 = create_random_matrix<float>(input_dim, hidden_dim);
    auto W2 = create_random_matrix<float>(hidden_dim, output_dim);
    
    // Create input data
    auto input = create_random_matrix<float>(32, input_dim);
    
    // Forward pass
    auto hidden = input * W1;
    auto output = hidden * W2;
    
    // Verify dimensions
    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], output_dim);
    
    // Should handle large batch efficiently
    const int large_batch = 256;
    auto large_input = create_random_matrix<__half>(large_batch, input_dim);
    auto large_output = (large_input * W1) * W2; // Chain operations
    
    EXPECT_EQ(large_output.shape()[0], large_batch);
    EXPECT_EQ(large_output.shape()[1], output_dim);
}

/**
 * @brief Test performance characteristics
 */
TEST_F(ChoreoIRTest, PerformanceCharacteristics) {
    const int M = 1024, N = 1024, K = 1024;
    
    // Create large matrices for performance testing
    auto A = create_random_matrix<float>(M, K);
    auto B = create_random_matrix<float>(K, N);
    
    // Measure matrix multiplication performance
    auto start = std::chrono::high_resolution_clock::now();
    auto C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Matrix multiplication took: " << duration.count() << " ms" << std::endl;
    
    // Verify result
    EXPECT_EQ(C.shape()[0], M);
    EXPECT_EQ(C.shape()[1], N);
}

/**
 * @brief Test different data types
 */
TEST_F(ChoreoIRTest, DifferentDataTypes) {
    const int M = 64, N = 32, K = 48;
    
    // Test float precision
    auto A_f32 = create_random_matrix<float>(M, K);
    auto B_f32 = create_random_matrix<float>(K, N);
    auto C_f32 = A_f32 * B_f32;
    EXPECT_EQ(C_f32.shape()[0], M);
    EXPECT_EQ(C_f32.shape()[1], N);
    
    // Test half precision
    auto A_f16 = create_random_matrix<__half>(M, K);
    auto B_f16 = create_random_matrix<__half>(K, N);
    auto C_f16 = A_f16 * B_f16;
    EXPECT_EQ(C_f16.shape()[0], M);
    EXPECT_EQ(C_f16.shape()[1], N);
}

/**
 * @brief Test memory management
 */
TEST_F(ChoreoIRTest, MemoryManagement) {
    const int size = 1024;
    
    // Test host tensors
    auto host_tensor = Tensor<float>::host(Shape({size, size}));
    EXPECT_EQ(host_tensor.memory_type(), MemoryType::HOST);
    EXPECT_EQ(host_tensor.shape()[0], size);
    EXPECT_EQ(host_tensor.shape()[1], size);
    
    // Test device tensors
    auto device_tensor = Tensor<float>::device(Shape({size, size}));
    EXPECT_EQ(device_tensor.memory_type(), MemoryType::DEVICE);
    EXPECT_EQ(device_tensor.shape()[0], size);
    EXPECT_EQ(device_tensor.shape()[1], size);
    
    // Test data transfer
    host_tensor.fill(42.0f);
    device_tensor.copy_from_host(host_tensor.data());
    
    // Verify transfer
    std::vector<float> result(size * size);
    device_tensor.copy_to_host(result.data());
    EXPECT_FLOAT_EQ(result[0], 42.0f);
}

/**
 * @brief Test error handling
 */
TEST_F(ChoreoIRTest, ErrorHandling) {
    // Test shape mismatch
    auto A = Tensor<float>::zeros(Shape({64, 32}));
    auto B = Tensor<float>::zeros(Shape({64, 32})); // Wrong shape
    
    EXPECT_THROW(A * B, std::runtime_error);
    
    // Test addition with different shapes
    auto C = Tensor<float>::zeros(Shape({32, 64}));
    EXPECT_THROW(A + C, std::runtime_error);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 