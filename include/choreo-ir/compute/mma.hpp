/**
 * @file mma.hpp
 * @brief Matrix Multiply Accumulate (MMA) operations
 */

#ifndef CHOREO_IR_COMPUTE_MMA_HPP
#define CHOREO_IR_COMPUTE_MMA_HPP

#include <cuda_runtime.h>
#include <mma.h>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "validation.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @struct MMAConfig
 * @brief Configuration for MMA operations
 */
struct MMAConfig {
    // MMA tile dimensions
    dim_t m = 16;  // Matrix A rows
    dim_t n = 16;  // Matrix B columns  
    dim_t k = 16;  // Matrix A columns / Matrix B rows
    
    // Data types
    using InputType = __half;
    using AccumulatorType = float;
    using OutputType = __half;
    
    MMAConfig() = default;
    MMAConfig(dim_t m, dim_t n, dim_t k) : m(m), n(n), k(k) {}
};

/**
 * @brief MMA GEMM: C = alpha * A * B + beta * C
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param alpha Scaling factor
 * @param beta Scaling factor
 * @param config MMA configuration
 */
template<typename T>
void mma_gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
              T alpha = T(1), T beta = T(0),
              const MMAConfig& config = MMAConfig());

/**
 * @brief MMA matrix multiplication: C = A * B
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param config MMA configuration
 */
template<typename T>
void mma_matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
                const MMAConfig& config = MMAConfig());

/**
 * @brief Check if MMA can be used for given tensors
 * @param A Input matrix A
 * @param B Input matrix B
 * @return true if MMA can be used
 */
template<typename T>
bool can_use_mma(const Tensor<T>& A, const Tensor<T>& B);

/**
 * @brief Get optimal MMA configuration for tensors
 * @param A Input matrix A
 * @param B Input matrix B
 * @return Optimal MMA configuration
 */
template<typename T>
MMAConfig get_optimal_mma_config(const Tensor<T>& A, const Tensor<T>& B);

/**
 * @brief Validate MMA contract requirements
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param config MMA configuration
 */
template<typename T>
void validate_mma_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                          const MMAConfig& config = MMAConfig());

/**
 * @brief MMA 8x8x4, int8, row_major micro-kernel (single tile, no tiling/loop)
 * @param A Pointer to 8x4 int8 tile (row-major)
 * @param B Pointer to 4x8 int8 tile (row-major)
 * @param C Pointer to 8x8 int32 tile (row-major, accumulator)
 * @param lda Leading dimension of A
 * @param ldb Leading dimension of B
 * @param ldc Leading dimension of C
 * @note No tiling, no loop, only one tile. User must ensure shape and alignment.
 */
__device__ inline void mma_tile_gemm_8x8x4_int8_row_major(
    const int8_t* A, const int8_t* B, int32_t* C, int lda, int ldb, int ldc) {
#if (__CUDA_ARCH__ >= 750)
    using namespace nvcuda;
    mma::fragment<mma::matrix_a, 8, 8, 4, int8_t, mma::row_major> a_frag;
    mma::fragment<mma::matrix_b, 8, 8, 4, int8_t, mma::row_major> b_frag;
    mma::fragment<mma::accumulator, 8, 8, 4, int32_t> c_frag;
    mma::load_matrix_sync(a_frag, A, lda);
    mma::load_matrix_sync(b_frag, B, ldb);
    mma::fill_fragment(c_frag, 0);
    mma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    mma::store_matrix_sync(C, c_frag, ldc, mma::mem_row_major);
#else
    // Not supported on this architecture
#endif
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_MMA_HPP 