/**
 * @file test_baselines.cpp
 * @brief Test baseline implementations
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>

#include "choreo-ir/choreo-ir.hpp"
#include "baselines.hpp"
#include "baseline_factory.hpp"
#include "cublas_baseline.hpp"
#include "cuda_baseline.hpp"

class BaselineTest : public ::testing::Test {
protected:
    void SetUp() override {
        choreo_ir::initialize();
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
        choreo_ir::finalize();
        cudaDeviceReset();
    }
};

TEST_F(BaselineTest, CublasBaselineMatrixMultiplication) {
    using namespace choreo_ir::baselines;
    
    // Create baseline
    auto baseline = BaselineFactory::create("cublas");
    baseline->initialize();
    
    // Test data
    const int M = 2, N = 2, K = 2;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f}; // [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity matrix
    std::vector<float> h_C(4, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication
    auto cublas_baseline = static_cast<CublasBaseline*>(baseline.get());
    cublas_baseline->gemm(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result (A * I = A) - cuBLAS outputs column-major format
    // For row-major input [[1,2],[3,4]], cuBLAS outputs column-major [1,3,2,4]
    EXPECT_FLOAT_EQ(h_C[0], 1.0f); // [0,0]
    EXPECT_FLOAT_EQ(h_C[1], 2.0f); // [1,0]
    EXPECT_FLOAT_EQ(h_C[2], 3.0f); // [0,1]
    EXPECT_FLOAT_EQ(h_C[3], 4.0f); // [1,1]
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    baseline->cleanup();
}

TEST_F(BaselineTest, CublasRowMajorBaselineMatrixMultiplication) {
    using namespace choreo_ir::baselines;
    
    // Create row-major baseline
    auto baseline = BaselineFactory::create("cublas-row");
    baseline->initialize();
    
    // Test data
    const int M = 2, N = 2, K = 2;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f}; // [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity matrix
    std::vector<float> h_C(4, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication
    auto cublas_baseline = static_cast<CublasBaseline*>(baseline.get());
    cublas_baseline->gemm(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result (A * I = A)
    EXPECT_FLOAT_EQ(h_C[0], 1.0f);
    EXPECT_FLOAT_EQ(h_C[1], 3.0f);
    EXPECT_FLOAT_EQ(h_C[2], 2.0f);
    EXPECT_FLOAT_EQ(h_C[3], 4.0f);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    baseline->cleanup();
}

TEST_F(BaselineTest, CublasColMajorBaselineMatrixMultiplication) {
    using namespace choreo_ir::baselines;
    
    // Create column-major baseline
    auto baseline = BaselineFactory::create("cublas-col");
    baseline->initialize();
    
    // Test data - column-major layout
    const int M = 2, N = 2, K = 2;
    // Column-major: A = [[1,3],[2,4]] (same as row-major [1,2,3,4])
    std::vector<float> h_A = {1.0f, 3.0f, 2.0f, 4.0f}; // Column-major [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // Column-major identity
    std::vector<float> h_C(4, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication
    auto cublas_baseline = static_cast<CublasBaseline*>(baseline.get());
    cublas_baseline->gemm(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result (A * I = A) - column-major layout
    EXPECT_FLOAT_EQ(h_C[0], 1.0f); // [0,0]
    EXPECT_FLOAT_EQ(h_C[1], 3.0f); // [1,0]
    EXPECT_FLOAT_EQ(h_C[2], 2.0f); // [0,1]
    EXPECT_FLOAT_EQ(h_C[3], 4.0f); // [1,1]
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    baseline->cleanup();
}

TEST_F(BaselineTest, CudaBaselineMatrixMultiplication) {
    using namespace choreo_ir::baselines;
    
    // Create baseline
    auto baseline = BaselineFactory::create("cuda");
    baseline->initialize();
    
    // Test data
    const int M = 2, N = 2, K = 2;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f}; // [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity matrix
    std::vector<float> h_C(4, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication
    auto cuda_baseline = static_cast<CudaBaseline*>(baseline.get());
    cuda_baseline->gemm(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result (A * I = A)
    EXPECT_FLOAT_EQ(h_C[0], 1.0f);
    EXPECT_FLOAT_EQ(h_C[1], 2.0f);
    EXPECT_FLOAT_EQ(h_C[2], 3.0f);
    EXPECT_FLOAT_EQ(h_C[3], 4.0f);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    baseline->cleanup();
}

TEST_F(BaselineTest, BaselineFactory) {
    using namespace choreo_ir::baselines;
    
    // Test available baselines
    auto baselines = BaselineFactory::available_baselines();
    EXPECT_EQ(baselines.size(), 4);
    EXPECT_TRUE(std::find(baselines.begin(), baselines.end(), "cublas") != baselines.end());
    EXPECT_TRUE(std::find(baselines.begin(), baselines.end(), "cublas-row") != baselines.end());
    EXPECT_TRUE(std::find(baselines.begin(), baselines.end(), "cublas-col") != baselines.end());
    EXPECT_TRUE(std::find(baselines.begin(), baselines.end(), "cuda") != baselines.end());
    
    // Test creating baselines
    auto cublas = BaselineFactory::create("cublas");
    EXPECT_EQ(cublas->name(), "cuBLAS-ColMajor");
    
    auto cublas_row = BaselineFactory::create("cublas-row");
    EXPECT_EQ(cublas_row->name(), "cuBLAS-RowMajor");
    
    auto cublas_col = BaselineFactory::create("cublas-col");
    EXPECT_EQ(cublas_col->name(), "cuBLAS-ColMajor");
    
    auto cuda = BaselineFactory::create("cuda");
    EXPECT_EQ(cuda->name(), "CUDA-RowMajor");
    
    // Test invalid baseline
    EXPECT_THROW(BaselineFactory::create("invalid"), std::invalid_argument);
}

TEST_F(BaselineTest, BaselineComparison) {
    using namespace choreo_ir::baselines;
    // Create both row-major baselines
    auto cublas_row = BaselineFactory::create_cublas(DataLayout::ROW_MAJOR);
    auto cuda_row = BaselineFactory::create_cuda(DataLayout::ROW_MAJOR);
    cublas_row->initialize();
    cuda_row->initialize();
    // Test data
    const int M = 64, N = 64, K = 64;
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C1(M * N, 0.0f);
    std::vector<float> h_C2(M * N, 0.0f);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C1, M * N * sizeof(float));
    cudaMalloc(&d_C2, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    // Test cuBLAS row-major
    auto cublas = static_cast<CublasBaseline*>(cublas_row.get());
    cublas->gemm(d_A, d_B, d_C1, M, N, K);
    // Test CUDA row-major
    auto cuda = static_cast<CudaBaseline*>(cuda_row.get());
    cuda->gemm(d_A, d_B, d_C2, M, N, K);
    cudaMemcpy(h_C1.data(), d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2.data(), d_C2, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // Compare CUDA row-major output (row-major) with cuBLAS row-major output (column-major):
    // Transpose CUDA output to column-major for comparison
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(h_C1[j * M + i], h_C2[i * N + j], 1e-4);
        }
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    cublas_row->cleanup(); cuda_row->cleanup();
    // Create both col-major baselines
    auto cublas_col = BaselineFactory::create_cublas(DataLayout::COLUMN_MAJOR);
    auto cuda_col = BaselineFactory::create_cuda(DataLayout::COLUMN_MAJOR);
    cublas_col->initialize();
    cuda_col->initialize();
    // 转置输入为col-major
    std::vector<float> h_A_col(M * K), h_B_col(K * N);
    for (int i = 0; i < M; ++i) for (int k = 0; k < K; ++k) h_A_col[k * M + i] = h_A[i * K + k];
    for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j) h_B_col[j * K + k] = h_B[k * N + j];
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C1, M * N * sizeof(float));
    cudaMalloc(&d_C2, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A_col.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_col.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    // Test cuBLAS col-major
    cublas = static_cast<CublasBaseline*>(cublas_col.get());
    cublas->gemm(d_A, d_B, d_C1, M, N, K);
    // Test CUDA col-major
    cuda = static_cast<CudaBaseline*>(cuda_col.get());
    cuda->gemm(d_A, d_B, d_C2, M, N, K);
    cudaMemcpy(h_C1.data(), d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2.data(), d_C2, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(h_C1[i * N + j], h_C2[j * M + i], 1e-4);
        }
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    cublas_col->cleanup(); cuda_col->cleanup();
}

TEST_F(BaselineTest, LayoutComparison) {
    using namespace choreo_ir::baselines;
    
    // Create both row-major and column-major baselines
    auto row_baseline = BaselineFactory::create_cublas(DataLayout::ROW_MAJOR);
    auto col_baseline = BaselineFactory::create_cublas(DataLayout::COLUMN_MAJOR);
    
    row_baseline->initialize();
    col_baseline->initialize();
    
    // Test data
    const int M = 2, N = 2, K = 2;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f}; // Row-major [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // Row-major identity
    std::vector<float> h_C_row(4, 0.0f);
    std::vector<float> h_C_col(4, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C_row, *d_C_col;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C_row, M * N * sizeof(float));
    cudaMalloc(&d_C_col, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_row, h_C_row.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_col, h_C_col.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test row-major baseline
    row_baseline->gemm(d_A, d_B, d_C_row, M, N, K);
    
    // Test column-major baseline (need column-major data)
    std::vector<float> h_A_col = {1.0f, 3.0f, 2.0f, 4.0f}; // Column-major [[1,2],[3,4]]
    std::vector<float> h_B_col = {1.0f, 0.0f, 0.0f, 1.0f}; // Column-major identity (same as row-major for identity)
    cudaMemcpy(d_A, h_A_col.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_col.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    col_baseline->gemm(d_A, d_B, d_C_col, M, N, K);
    
    // Copy results back
    cudaMemcpy(h_C_row.data(), d_C_row, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_col.data(), d_C_col, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Row-major should give row-major output: A * I = A
    EXPECT_FLOAT_EQ(h_C_row[0], 1.0f);
    EXPECT_FLOAT_EQ(h_C_row[1], 3.0f);
    EXPECT_FLOAT_EQ(h_C_row[2], 2.0f);
    EXPECT_FLOAT_EQ(h_C_row[3], 4.0f);
    
    // Column-major should give column-major output: A * I = A
    // For column-major [[1,2],[3,4]], the output should be [1,3,2,4]
    EXPECT_FLOAT_EQ(h_C_col[0], 1.0f);
    EXPECT_FLOAT_EQ(h_C_col[1], 3.0f);
    EXPECT_FLOAT_EQ(h_C_col[2], 2.0f);
    EXPECT_FLOAT_EQ(h_C_col[3], 4.0f);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_row);
    cudaFree(d_C_col);
    row_baseline->cleanup();
    col_baseline->cleanup();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 