/**
 * @file compute_impl.hpp
 * @brief Implementation of compute operations with automatic tensor core selection
 */

#ifndef CHOREO_IR_COMPUTE_COMPUTE_IMPL_HPP
#define CHOREO_IR_COMPUTE_COMPUTE_IMPL_HPP

#include "compute.hpp"
#include "wmma.hpp"
#include "mma.hpp"
#include "cuda_core.hpp"
#include "validation.hpp"
#include "../utils/cuda_utils.hpp"

namespace choreo_ir {
namespace compute {

template<typename T>
void gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
          T alpha, T beta, const GEMMConfig& config) {
    // Validate contract
    validate_matmul_contract(A, B, C, "GEMM operation");
    
    // Get optimal configuration
    auto optimal_config = get_optimal_gemm_config(A, B);
    
    // Choose backend based on configuration and tensor properties
    if (config.backend == ComputeBackend::WMMA && can_use_wmma_gemm(A, B)) {
        wmma_gemm(A, B, C, alpha, beta, WMMAConfig());
    } else if (config.backend == ComputeBackend::MMA && can_use_tensor_core_gemm(A, B)) {
        mma_gemm(A, B, C, alpha, beta, MMAConfig());
    } else {
        // Fall back to CUDA core implementation with warning
        fprintf(stderr, "[WARN] Tensor core not used, fallback to CUDA core\n");
        cuda_core_gemm(A, B, C, alpha, beta, CudaCoreConfig());
    }
}

template<typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B,
               T alpha, const GEMMConfig& config) {
    // Create output tensor
    Tensor<T> C(Shape({A.shape()[A.shape().ndims() - 2], B.shape()[B.shape().ndims() - 1]}), A.memory_type());
    
    // Perform GEMM
    gemm(A, B, C, alpha, T(0), config);
    
    return C;
}

template<typename T>
void matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
            const GEMMConfig& config) {
    // Validate contract
    validate_matmul_contract(A, B, C, "Matrix multiplication");
    
    // Use GEMM with alpha=1, beta=0
    gemm(A, B, C, T(1), T(0), config);
}

template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B,
                 const GEMMConfig& config) {
    // Create output tensor
    Tensor<T> C(Shape({A.shape()[A.shape().ndims() - 2], B.shape()[B.shape().ndims() - 1]}), A.memory_type());
    
    // Perform matrix multiplication
    matmul(A, B, C, config);
    
    return C;
}

template<typename T>
GEMMConfig get_optimal_gemm_config(const Tensor<T>& A, const Tensor<T>& B) {
    GEMMConfig config;
    
    // Check if tensor core can be used
    if (can_use_tensor_core_gemm(A, B)) {
        config.backend = ComputeBackend::MMA;
        config.use_tensor_core = true;
    } else if (can_use_wmma_gemm(A, B)) {
        config.backend = ComputeBackend::WMMA;
        config.use_tensor_core = false;
    } else {
        config.backend = ComputeBackend::CUDA_CORE;
        config.use_tensor_core = false;
    }
    
    return config;
}

template<typename T>
bool can_use_tensor_core_gemm(const Tensor<T>& A, const Tensor<T>& B) {
    // Tensor core requirements:
    // 1. Data type must be __half
    // 2. Matrix dimensions must be multiples of 16
    // 3. Both tensors must be on GPU
    
    if (!std::is_same_v<T, __half>) {
        return false;
    }
    
    if (A.memory_type() != MemoryType::DEVICE || B.memory_type() != MemoryType::DEVICE) {
        return false;
    }
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    
    index_t M = A_shape[A_shape.ndims() - 2];
    index_t K = A_shape[A_shape.ndims() - 1];
    index_t K2 = B_shape[B_shape.ndims() - 2];
    index_t N = B_shape[B_shape.ndims() - 1];
    
    // Check if dimensions are multiples of 16
    return (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
}

template<typename T>
bool can_use_wmma_gemm(const Tensor<T>& A, const Tensor<T>& B) {
    // WMMA requirements:
    // 1. Data type must be __half
    // 2. Matrix dimensions must be multiples of 16
    // 3. Both tensors must be on GPU
    
    if (!std::is_same_v<T, __half>) {
        return false;
    }
    
    if (A.memory_type() != MemoryType::DEVICE || B.memory_type() != MemoryType::DEVICE) {
        return false;
    }
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    
    index_t M = A_shape[A_shape.ndims() - 2];
    index_t K = A_shape[A_shape.ndims() - 1];
    index_t K2 = B_shape[B_shape.ndims() - 2];
    index_t N = B_shape[B_shape.ndims() - 1];
    
    // Check if dimensions are multiples of 16
    return (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_COMPUTE_IMPL_HPP 