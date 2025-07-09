/**
 * @file matmul_kernels.cu
 * @brief Matrix multiplication CUDA kernel implementations
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include "choreo-ir/compute/matmul.hpp"
#include "choreo-ir/utils/cuda_utils.hpp"
#include "choreo-ir/utils/debug.hpp"

namespace choreo_ir {
namespace kernels {

using namespace nvcuda;

/**
 * @brief Naive matrix multiplication kernel
 */
template<typename T>
__global__ void matmul_naive_kernel(const T* A, const T* B, T* C,
                                   index_t M, index_t N, index_t K,
                                   index_t lda, index_t ldb, index_t ldc,
                                   T alpha, T beta) {
    index_t row = blockIdx.y * blockDim.y + threadIdx.y;
    index_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = T(0);
        for (index_t k = 0; k < K; ++k) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * @brief Shared memory tiled matrix multiplication kernel
 */
template<typename T, int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_shared_kernel(const T* A, const T* B, T* C,
                                    index_t M, index_t N, index_t K,
                                    index_t lda, index_t ldb, index_t ldc,
                                    T alpha, T beta) {
    __shared__ T shared_A[TILE_M][TILE_K];
    __shared__ T shared_B[TILE_K][TILE_N];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    T sum = T(0);
    
    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; ++tile) {
        // Load data into shared memory
        int k_idx = tile * TILE_K + tx;
        if (row < M && k_idx < K) {
            shared_A[ty][tx] = A[row * lda + k_idx];
        } else {
            shared_A[ty][tx] = T(0);
        }
        
        k_idx = tile * TILE_K + ty;
        if (k_idx < K && col < N) {
            shared_B[ty][tx] = B[k_idx * ldb + col];
        } else {
            shared_B[ty][tx] = T(0);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_K; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * @brief Tensor core matrix multiplication kernel using WMMA
 */
template<typename T>
__global__ void matmul_wmma_kernel(const T* A, const T* B, T* C,
                                  index_t M, index_t N, index_t K,
                                  index_t lda, index_t ldb, index_t ldc,
                                  float alpha, float beta) {
    // WMMA fragment declarations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Warp and block indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds check
    if (warpM >= (M + 15) / 16 || warpN >= (N + 15) / 16) return;
    
    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Main computation loop
    for (int k = 0; k < K; k += 16) {
        int a_row = warpM * 16;
        int a_col = k;
        int b_row = k;
        int b_col = warpN * 16;
        
        // Bounds checking for fragment loads
        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // Load fragments
            wmma::load_matrix_sync(a_frag, A + a_row * lda + a_col, lda);
            wmma::load_matrix_sync(b_frag, B + b_row * ldb + b_col, ldb);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store fragment to output matrix
    int c_row = warpM * 16;
    int c_col = warpN * 16;
    if (c_row < M && c_col < N) {
        // Scale and store
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * c_frag.x[i];
        }
        wmma::store_matrix_sync(C + c_row * ldc + c_col, c_frag, ldc, wmma::layout_t::row_major);
    }
}

/**
 * @brief Vectorized matrix multiplication kernel for small matrices
 */
template<typename T>
__global__ void matmul_vectorized_kernel(const T* A, const T* B, T* C,
                                        index_t M, index_t N, index_t K,
                                        index_t lda, index_t ldb, index_t ldc,
                                        T alpha, T beta) {
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(T);
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    
    if (row < M && col < N) {
        T results[VEC_SIZE] = {0};
        
        for (int k = 0; k < K; ++k) {
            T a_val = A[row * lda + k];
            
            // Vectorized load of B
            if (col + VEC_SIZE <= N) {
                auto b_vec = reinterpret_cast<const float4*>(&B[k * ldb + col]);
                auto b_vals = reinterpret_cast<const T*>(b_vec);
                
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    results[v] += a_val * b_vals[v];
                }
            } else {
                // Handle boundary case
                for (int v = 0; v < VEC_SIZE && col + v < N; ++v) {
                    results[v] += a_val * B[k * ldb + col + v];
                }
            }
        }
        
        // Vectorized store to C
        #pragma unroll
        for (int v = 0; v < VEC_SIZE && col + v < N; ++v) {
            C[row * ldc + col + v] = alpha * results[v] + beta * C[row * ldc + col + v];
        }
    }
}

/**
 * @brief Cooperative groups matrix multiplication kernel
 */
template<typename T>
__global__ void matmul_coop_kernel(const T* A, const T* B, T* C,
                                  index_t M, index_t N, index_t K,
                                  index_t lda, index_t ldb, index_t ldc,
                                  T alpha, T beta) {
    // Implementation using cooperative groups for better sync
    // This is a placeholder - full implementation would use cooperative_groups
    matmul_shared_kernel<T, 32, 32, 32><<<gridDim, blockDim>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
}

// Explicit template instantiations
template __global__ void matmul_naive_kernel<float>(const float*, const float*, float*,
                                                   index_t, index_t, index_t,
                                                   index_t, index_t, index_t,
                                                   float, float);

template __global__ void matmul_shared_kernel<float, 16, 16, 16>(const float*, const float*, float*,
                                                                index_t, index_t, index_t,
                                                                index_t, index_t, index_t,
                                                                float, float);

template __global__ void matmul_wmma_kernel<__half>(const __half*, const __half*, __half*,
                                                   index_t, index_t, index_t,
                                                   index_t, index_t, index_t,
                                                   float, float);

template __global__ void matmul_vectorized_kernel<float>(const float*, const float*, float*,
                                                        index_t, index_t, index_t,
                                                        index_t, index_t, index_t,
                                                        float, float);

} // namespace kernels
} // namespace choreo_ir 