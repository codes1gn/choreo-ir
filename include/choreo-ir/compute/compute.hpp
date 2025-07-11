/**
 * @file compute.hpp
 * @brief Core compute operations contract for GEMM/MACC patterns
 */

#ifndef CHOREO_IR_COMPUTE_COMPUTE_HPP
#define CHOREO_IR_COMPUTE_COMPUTE_HPP

#include <cuda_runtime.h>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"
#include "validation.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @enum ComputeBackend
 * @brief Available compute backends for GEMM operations
 */
enum class ComputeBackend {
    CUDA_CORE,  // CUDA core implementation
    WMMA,       // Warp Matrix Multiply Accumulate
    MMA         // Matrix Multiply Accumulate (tensor core)
};

/**
 * @struct GEMMConfig
 * @brief Configuration for GEMM operations
 */
struct GEMMConfig {
    ComputeBackend backend = ComputeBackend::CUDA_CORE;
    bool use_tensor_core = true;
    cudaStream_t stream = 0;
    
    // Tile dimensions for optimization
    dim_t tile_m = 32;
    dim_t tile_n = 32;
    dim_t tile_k = 32;
    
    GEMMConfig() = default;
    explicit GEMMConfig(ComputeBackend backend) : backend(backend) {}
};

/**
 * @brief GEMM operation: C = alpha * A * B + beta * C
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param alpha Scaling factor for A * B
 * @param beta Scaling factor for C
 * @param config GEMM configuration
 */
template<typename T>
void gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
          T alpha = T(1), T beta = T(0),
          const GEMMConfig& config = GEMMConfig());

/**
 * @brief GEMM with automatic output allocation: C = alpha * A * B
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param alpha Scaling factor
 * @param config GEMM configuration
 * @return Output matrix C (M x N)
 */
template<typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B,
               T alpha = T(1), const GEMMConfig& config = GEMMConfig());

/**
 * @brief Matrix multiplication: C = A * B (convenience wrapper)
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param config GEMM configuration
 */
template<typename T>
void matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
            const GEMMConfig& config = GEMMConfig());

/**
 * @brief Matrix multiplication with automatic output allocation
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param config GEMM configuration
 * @return Output matrix C (M x N)
 */
template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B,
                 const GEMMConfig& config = GEMMConfig());

/**
 * @brief Get optimal GEMM configuration for tensors
 * @param A Input matrix A
 * @param B Input matrix B
 * @return Optimal GEMM configuration
 */
template<typename T>
GEMMConfig get_optimal_gemm_config(const Tensor<T>& A, const Tensor<T>& B);

/**
 * @brief Check if tensor core can be used for GEMM
 * @param A Input matrix A
 * @param B Input matrix B
 * @return true if tensor core can be used
 */
template<typename T>
bool can_use_tensor_core_gemm(const Tensor<T>& A, const Tensor<T>& B);

/**
 * @brief Check if WMMA can be used for GEMM
 * @param A Input matrix A
 * @param B Input matrix B
 * @return true if WMMA can be used
 */
template<typename T>
bool can_use_wmma_gemm(const Tensor<T>& A, const Tensor<T>& B);

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_COMPUTE_HPP 