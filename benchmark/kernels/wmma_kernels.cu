// Copyright (c) 2024 Choreo-IR
//
// @file wmma_kernels.cu
// @brief WMMA (Tensor Core) micro-kernel implementations for all official tile sizes, dtypes, layouts.
//        Strictly use nvcuda::wmma API. This file will be extended to cover all combinations.

#include <mma.h>
#include <cuda_fp16.h>
#include "choreo-ir/tensor/tensor.hpp"
#include "choreo-ir/compute/wmma.hpp"
#include <cstdio>

using namespace nvcuda;

namespace choreo_ir {
namespace compute {

// Example: 16x16x16 __half row_major GEMM micro-kernel
__global__ void wmma_gemm_16x16x16_half_row_major_kernel(const __half* A, const __half* B, float* C,
                                                         int M, int N, int K, float alpha, float beta) {
    // Each warp computes one 16x16 tile of C
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 16;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    if (warpM * 16 >= M || warpN * 16 >= N) return;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K
    for (int k = 0; k < K; k += 16) {
        // Bounds check for K
        int a_row = warpM * 16;
        int a_col = k;
        int b_row = k;
        int b_col = warpN * 16;
        if (a_col + 16 > K || b_row + 16 > K) continue;

        // Load the inputs
        const __half* a_tile = A + a_row * K + a_col;
        const __half* b_tile = B + b_row * N + b_col;
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);

        // Perform the matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result
    float* c_tile = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

// TODO: Add more kernels for all tile sizes, dtypes, layouts as per NVIDIA doc

} // namespace compute
} // namespace choreo_ir 