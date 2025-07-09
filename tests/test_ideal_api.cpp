/**
 * @file test_ideal_api.cpp
 * @brief Unit tests for the ideal Choreo-IR programming experience
 * 
 * Tests the core programming patterns:
 * 1. Natural tensor creation and manipulation
 * 2. Intuitive data movement: dst = src.tile(shape)
 * 3. Simple tensor core usage: C = A * B
 * 4. Automatic configuration and memory management
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <chrono>

// Include our ideal API (will be implemented progressively)
#include "choreo-ir/ideal_api.hpp"

using namespace choreo_ir;

class IdealAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context
        CUDA_CHECK(cudaSetDevice(0));
        
        // Initialize cuBLAS for baseline comparisons
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
        // Set random seed for reproducible tests
        std::srand(42);
    }
    
    void TearDown() override {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
    
    cublasHandle_t cublas_handle;
    
    // Helper to create random matrices
    template<typename T>
    host_tensor<T> create_random_matrix(int M, int N, T min_val = T(-1), T max_val = T(1)) {
        auto tensor = host_tensor<T>::zeros({M, N});
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed
        
        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(min_val, max_val);
            for (int i = 0; i < M * N; ++i) {
                tensor.data()[i] = dis(gen);
            }
        } else if constexpr (std::is_same_v<T, __half>) {
            std::uniform_real_distribution<float> dis(float(min_val), float(max_val));
            for (int i = 0; i < M * N; ++i) {
                tensor.data()[i] = __float2half(dis(gen));
            }
        }
        
        return tensor;
    }
    
    // Helper to compare tensors with tolerance
    template<typename T>
    bool tensors_equal(const host_tensor<T>& a, const host_tensor<T>& b, T tolerance = T(1e-5)) {
        if (a.shape() != b.shape()) return false;
        
        for (size_t i = 0; i < a.numel(); ++i) {
            T diff = std::abs(a.data()[i] - b.data()[i]);
            if (diff > tolerance) {
                std::cout << "Mismatch at index " << i << ": " 
                         << a.data()[i] << " vs " << b.data()[i] 
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
TEST_F(IdealAPITest, TensorCreationAndProperties) {
    // Test different ways to create tensors
    auto A1 = host_tensor<float>::zeros({64, 32});
    auto A2 = host_tensor<float>::ones({64, 32});
    auto A3 = host_tensor<float>::random({64, 32});
    
    // Check shapes
    EXPECT_EQ(A1.shape(), std::make_tuple(64, 32));
    EXPECT_EQ(A1.numel(), 64 * 32);
    EXPECT_EQ(A1.ndims(), 2);
    
    // Check initialization
    EXPECT_FLOAT_EQ(A1.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(A2.data()[0], 1.0f);
    
    // Test device transfer
    auto d_A = A1.to_device();
    EXPECT_EQ(d_A.shape(), A1.shape());
    EXPECT_TRUE(d_A.is_on_device());
    
    auto h_A = d_A.to_host();
    EXPECT_TRUE(tensors_equal(A1, h_A));
}

/**
 * @brief Test the core programming pattern: C = A * B
 */
TEST_F(IdealAPITest, SimpleMatrixMultiplication) {
    const int M = 128, N = 128, K = 128;
    
    // Create random matrices
    auto A = create_random_matrix<float>(M, K);
    auto B = create_random_matrix<float>(K, N);
    
    // The ideal experience: just like math!
    auto C = A * B;
    
    // Verify shape
    EXPECT_EQ(C.shape(), std::make_tuple(M, N));
    
    // Compare with cuBLAS baseline
    auto C_cublas = host_tensor<float>::zeros({M, N});
    auto d_A = A.to_device();
    auto d_B = B.to_device();
    auto d_C = C_cublas.to_device();
    
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha,
                           d_B.data(), N,
                           d_A.data(), K,
                           &beta, d_C.data(), N));
    
    C_cublas = d_C.to_host();
    
    // Should match cuBLAS within tolerance
    EXPECT_TRUE(tensors_equal(C, C_cublas, 1e-4f));
}

/**
 * @brief Test mixed precision matrix multiplication
 */
TEST_F(IdealAPITest, MixedPrecisionMatmul) {
    const int M = 256, N = 256, K = 256;
    
    // Input in half precision
    auto A = create_random_matrix<__half>(M, K);
    auto B = create_random_matrix<__half>(K, N);
    
    // Output in float (automatic promotion)
    auto C = host_tensor<float>::zeros({M, N});
    matmul(A, B, C); // Should use tensor cores automatically
    
    EXPECT_EQ(C.shape(), std::make_tuple(M, N));
    
    // Verify that tensor cores were used (check performance)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        matmul(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should be fast enough to indicate tensor core usage
    double tflops = (2.0 * M * N * K * 10) / (duration.count() * 1e-6) / 1e12;
    std::cout << "Mixed precision GEMM: " << tflops << " TFLOPS" << std::endl;
    
    // On modern GPUs with tensor cores, should achieve > 50 TFLOPS
    // This is a rough check - actual threshold depends on hardware
    EXPECT_GT(tflops, 1.0); // Very conservative check
}

/**
 * @brief Test different matrix sizes and shapes
 */
TEST_F(IdealAPITest, VariousMatrixSizes) {
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {64, 64, 64},       // Small
        {128, 256, 128},    // Rectangular
        {512, 512, 512},    // Medium square
        {1024, 128, 256},   // Tall and narrow
        {16, 16, 16},       // Tensor core minimum
        {32, 32, 32},       // Tensor core aligned
    };
    
    for (auto [M, N, K] : test_sizes) {
        SCOPED_TRACE("Testing size: " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K));
        
        auto A = create_random_matrix<float>(M, K);
        auto B = create_random_matrix<float>(K, N);
        
        // Should work regardless of size
        auto C = A * B;
        EXPECT_EQ(C.shape(), std::make_tuple(M, N));
        
        // Basic sanity check: C should not be all zeros
        bool has_nonzero = false;
        for (size_t i = 0; i < C.numel(); ++i) {
            if (std::abs(C.data()[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero);
    }
}

/**
 * @brief Test batch matrix multiplication
 */
TEST_F(IdealAPITest, BatchMatrixMultiplication) {
    const int batch_size = 8;
    const int M = 64, N = 64, K = 64;
    
    // Create batch tensors (batch_size, M, N)
    auto A = host_tensor<float>::random({batch_size, M, K});
    auto B = host_tensor<float>::random({batch_size, K, N});
    
    // Batch multiplication should be natural
    auto C = batch_matmul(A, B);
    
    EXPECT_EQ(C.shape(), std::make_tuple(batch_size, M, N));
    
    // Verify by computing individual multiplications
    for (int b = 0; b < batch_size; ++b) {
        auto A_slice = A.slice(b, b + 1).squeeze(0); // Remove batch dim
        auto B_slice = B.slice(b, b + 1).squeeze(0);
        auto C_expected = A_slice * B_slice;
        auto C_slice = C.slice(b, b + 1).squeeze(0);
        
        EXPECT_TRUE(tensors_equal(C_slice, C_expected, 1e-5f));
    }
}

/**
 * @brief Test convolution operation
 */
TEST_F(IdealAPITest, BasicConvolution) {
    const int N = 2, C = 64, H = 32, W = 32;
    const int K = 128, R = 3, S = 3;
    
    // Create input and weight tensors
    auto input = host_tensor<float>::random({N, C, H, W});
    auto weight = host_tensor<float>::random({K, C, R, S});
    
    // Convolution should be simple
    auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
    
    // Check output shape (with padding=1, size should be preserved)
    EXPECT_EQ(output.shape(), std::make_tuple(N, K, H, W));
}

/**
 * @brief Test automatic memory management
 */
TEST_F(IdealAPITest, AutomaticMemoryManagement) {
    const int M = 512, N = 512, K = 512;
    
    {
        // Create large tensors in nested scope
        auto A = host_tensor<float>::random({M, K});
        auto B = host_tensor<float>::random({K, N});
        auto C = A * B;
        
        EXPECT_EQ(C.shape(), std::make_tuple(M, N));
    } // Tensors should be automatically cleaned up here
    
    // Should still be able to create new tensors without memory issues
    auto A2 = host_tensor<float>::random({M, K});
    auto B2 = host_tensor<float>::random({K, N});
    auto C2 = A2 * B2;
    
    EXPECT_EQ(C2.shape(), std::make_tuple(M, N));
}

/**
 * @brief Test error handling and edge cases
 */
TEST_F(IdealAPITest, ErrorHandlingAndEdgeCases) {
    // Mismatched dimensions should throw or handle gracefully
    auto A = host_tensor<float>::random({64, 32});
    auto B = host_tensor<float>::random({16, 64}); // Wrong K dimension
    
    // This should either throw an exception or handle gracefully
    EXPECT_THROW({
        auto C = A * B;
    }, std::runtime_error);
    
    // Zero-sized tensors
    auto empty_A = host_tensor<float>::zeros({0, 0});
    EXPECT_EQ(empty_A.numel(), 0);
    
    // Single element tensors
    auto scalar_A = host_tensor<float>::ones({1, 1});
    auto scalar_B = host_tensor<float>::ones({1, 1});
    auto scalar_C = scalar_A * scalar_B;
    
    EXPECT_EQ(scalar_C.shape(), std::make_tuple(1, 1));
    EXPECT_FLOAT_EQ(scalar_C.data()[0], 1.0f);
}

/**
 * @brief Performance regression test
 */
TEST_F(IdealAPITest, PerformanceRegression) {
    const int M = 1024, N = 1024, K = 1024;
    const int num_iterations = 10;
    
    auto A = create_random_matrix<float>(M, K);
    auto B = create_random_matrix<float>(K, N);
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        auto C = A * B;
    }
    
    // Measure our implementation
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto C = A * B;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto our_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Measure cuBLAS baseline
    auto d_A = A.to_device();
    auto d_B = B.to_device();
    auto d_C = host_tensor<float>::zeros({M, N}).to_device();
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warm up cuBLAS
    for (int i = 0; i < 3; ++i) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               d_B.data(), N,
                               d_A.data(), K,
                               &beta, d_C.data(), N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               d_B.data(), N,
                               d_A.data(), K,
                               &beta, d_C.data(), N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    auto cublas_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate TFLOPS
    double ops = 2.0 * M * N * K * num_iterations;
    double our_tflops = ops / (our_duration.count() * 1e-6) / 1e12;
    double cublas_tflops = ops / (cublas_duration.count() * 1e-6) / 1e12;
    
    std::cout << "Our implementation: " << our_tflops << " TFLOPS" << std::endl;
    std::cout << "cuBLAS baseline: " << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "Performance ratio: " << (our_tflops / cublas_tflops * 100) << "%" << std::endl;
    
    // We should achieve at least 70% of cuBLAS performance
    EXPECT_GT(our_tflops, cublas_tflops * 0.7);
}

/**
 * @brief Integration test: Real-world usage pattern
 */
TEST_F(IdealAPITest, RealWorldUsagePattern) {
    // Simulate a small neural network layer
    const int batch_size = 32;
    const int input_dim = 512;
    const int hidden_dim = 1024;
    const int output_dim = 256;
    
    // Input and weights
    auto input = host_tensor<__half>::random({batch_size, input_dim});
    auto W1 = host_tensor<__half>::random({input_dim, hidden_dim});
    auto W2 = host_tensor<__half>::random({hidden_dim, output_dim});
    auto bias1 = host_tensor<__half>::random({hidden_dim});
    auto bias2 = host_tensor<__half>::random({output_dim});
    
    // Forward pass should be natural
    auto hidden = (input * W1) + bias1;  // Broadcasting should work
    // auto hidden_relu = relu(hidden);      // Element-wise operations
    auto output = (hidden * W2) + bias2;
    
    EXPECT_EQ(output.shape(), std::make_tuple(batch_size, output_dim));
    
    // Should handle large batch efficiently
    const int large_batch = 256;
    auto large_input = host_tensor<__half>::random({large_batch, input_dim});
    auto large_output = (large_input * W1) * W2; // Chain operations
    
    EXPECT_EQ(large_output.shape(), std::make_tuple(large_batch, output_dim));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA devices found. Skipping GPU tests." << std::endl;
        return 0;
    }
    
    return RUN_ALL_TESTS();
} 