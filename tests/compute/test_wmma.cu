// Copyright (c) 2024 Choreo-IR
//
// @file test_wmma.cu
// @brief Unit tests for WMMA (Tensor Core) micro-kernel (16x16x16, __half, row_major)

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include "choreo-ir/compute/wmma.hpp"
#include "choreo-ir/compute/cuda_core.hpp"
#include "choreo-ir/tensor/tensor.hpp"
#include <vector>
#include <random>

using namespace choreo_ir;
using namespace choreo_ir::compute;

namespace {

// Helper: fill host vector with random half values
void fill_random_half(std::vector<__half>& vec) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& h : vec) h = __float2half(dist(gen));
}

__global__ void test_kernel(const __half* A, const __half* B, float* C, int lda, int ldb, int ldc) {
    // Only one warp/thread block for single tile
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        choreo_ir::compute::wmma_tile_gemm_16x16x16_half_row_major(A, B, C, lda, ldb, ldc);
    }
}

TEST(WMMATileTest, SingleTile16x16x16HalfRowMajor) {
    constexpr int M = 16, N = 16, K = 16;
    std::vector<__half> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& h : h_A) h = __float2half(dist(gen));
    for (auto& h : h_B) h = __float2half(dist(gen));
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));
    test_kernel<<<1, 32>>>(d_A, d_B, d_C, K, N, N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // Reference: host GEMM
    std::vector<float> ref_C(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                ref_C[m * N + n] += __half2float(h_A[m * K + k]) * __half2float(h_B[k * N + n]);
    for (int i = 0; i < M * N; ++i)
        ASSERT_NEAR(h_C[i], ref_C[i], 1e-2f);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

} // namespace 