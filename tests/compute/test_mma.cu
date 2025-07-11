// Copyright (c) 2024 Choreo-IR
//
// @file test_mma.cu
// @brief Unit tests for MMA (Tensor Core) micro-kernel (8x8x4, int8, row_major)

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "choreo-ir/compute/mma.hpp"
#include <vector>
#include <random>
#include <cstdint>

using namespace choreo_ir;
using namespace choreo_ir::compute;

__global__ void test_kernel(const int8_t* A, const int8_t* B, int32_t* C, int lda, int ldb, int ldc) {
#if (__CUDA_ARCH__ >= 750)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mma_tile_gemm_8x8x4_int8_row_major(A, B, C, lda, ldb, ldc);
    }
#endif
}

TEST(MMATileTest, SingleTile8x8x4Int8RowMajor) {
    constexpr int M = 8, N = 8, K = 4;
    std::vector<int8_t> h_A(M * K), h_B(K * N);
    std::vector<int32_t> h_C(M * N, 0);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-8, 8);
    for (auto& h : h_A) h = static_cast<int8_t>(dist(gen));
    for (auto& h : h_B) h = static_cast<int8_t>(dist(gen));
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(int32_t));
    test_kernel<<<1, 32>>>(d_A, d_B, d_C, K, N, N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    // Reference: host GEMM
    std::vector<int32_t> ref_C(M * N, 0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                ref_C[m * N + n] += static_cast<int32_t>(h_A[m * K + k]) * static_cast<int32_t>(h_B[k * N + n]);
    for (int i = 0; i < M * N; ++i)
        ASSERT_EQ(h_C[i], ref_C[i]);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
} 