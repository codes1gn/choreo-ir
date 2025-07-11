/**
 * @file cuda_core.hpp
 * @brief CUDA core GEMM operations with micro-kernel implementation
 */

#ifndef CHOREO_IR_COMPUTE_CUDA_CORE_HPP
#define CHOREO_IR_COMPUTE_CUDA_CORE_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "validation.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @struct CudaCoreConfig
 * @brief Configuration for CUDA core operations
 */
struct CudaCoreConfig {
    // Tile dimensions for optimization
    dim_t tile_m = 32;
    dim_t tile_n = 32;
    dim_t tile_k = 32;
    
    // Thread block dimensions
    dim_t block_m = 16;
    dim_t block_n = 16;
    dim_t block_k = 16;
    
    // Shared memory usage
    size_t shared_memory_size = 0;
    
    CudaCoreConfig() = default;
    CudaCoreConfig(dim_t tile_m, dim_t tile_n, dim_t tile_k) 
        : tile_m(tile_m), tile_n(tile_n), tile_k(tile_k) {}
};

/**
 * @brief CUDA core micro-kernel for GEMM computation
 * @tparam T Data type (float, __half, etc.)
 * @param A Input matrix A
 * @param B Input matrix B  
 * @param C Output matrix C
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A * B
 * @param beta Scaling factor for C
 */
template<typename T>
__device__ __forceinline__ void cuda_core_micro_kernel(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    dim_t M, dim_t N, dim_t K, T alpha, T beta) {
    
    // Get thread indices
    dim_t tx = threadIdx.x;
    dim_t ty = threadIdx.y;
    dim_t bx = blockIdx.x;
    dim_t by = blockIdx.y;
    
    // Calculate global indices
    dim_t row = by * blockDim.y + ty;
    dim_t col = bx * blockDim.x + tx;
    
    // Bounds check
    if (row >= M || col >= N) return;
    
    // Compute dot product
    T sum = T(0);
    for (dim_t k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // Apply scaling and accumulate
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

/**
 * @brief CUDA core GEMM kernel with shared memory optimization
 * @tparam T Data type
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor
 * @param beta Scaling factor
 */
template<typename T>
__global__ void cuda_core_gemm_kernel(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    dim_t M, dim_t N, dim_t K, T alpha, T beta) {
    
    // Shared memory for tiles
    extern __shared__ T shared_mem[];
    T* tile_A = shared_mem;
    T* tile_B = shared_mem + blockDim.x * blockDim.y;
    
    // Thread indices
    dim_t tx = threadIdx.x;
    dim_t ty = threadIdx.y;
    dim_t bx = blockIdx.x;
    dim_t by = blockIdx.y;
    
    // Global indices
    dim_t row = by * blockDim.y + ty;
    dim_t col = bx * blockDim.x + tx;
    
    // Accumulator
    T sum = T(0);
    
    // Iterate over K dimension in tiles
    for (dim_t k = 0; k < K; k += blockDim.x) {
        // Load tile A
        if (row < M && k + tx < K) {
            tile_A[ty * blockDim.x + tx] = A[row * K + (k + tx)];
        } else {
            tile_A[ty * blockDim.x + tx] = T(0);
        }
        
        // Load tile B
        if (k + ty < K && col < N) {
            tile_B[ty * blockDim.x + tx] = B[(k + ty) * N + col];
        } else {
            tile_B[ty * blockDim.x + tx] = T(0);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (dim_t i = 0; i < blockDim.x; ++i) {
            sum += tile_A[ty * blockDim.x + i] * tile_B[i * blockDim.x + tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/**
 * @brief CUDA core GEMM: C = alpha * A * B + beta * C
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param alpha Scaling factor
 * @param beta Scaling factor
 * @param config CUDA core configuration
 */
template<typename T>
void cuda_core_gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
                    T alpha, T beta, const CudaCoreConfig& config) {
    // Validate contract
    validate_cuda_core_contract(A, B, C, config);
    
    // Get tensor dimensions
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    
    dim_t M = A_shape[A_shape.ndims() - 2];
    dim_t K = A_shape[A_shape.ndims() - 1];
    dim_t K2 = B_shape[B_shape.ndims() - 2];
    dim_t N = B_shape[B_shape.ndims() - 1];
    
    // Verify dimensions match
    if (K != K2) {
        throw std::runtime_error("Matrix dimensions do not match for GEMM");
    }
    
    // Calculate grid and block dimensions
    dim3 block_dim(config.block_m, config.block_n);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                   (M + block_dim.y - 1) / block_dim.y);
    
    // Calculate shared memory size
    size_t shared_mem_size = (config.block_m * config.block_k + 
                              config.block_k * config.block_n) * sizeof(T);
    
    // Launch kernel
    cuda_core_gemm_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        A.data(), B.data(), C.data(), M, N, K, alpha, beta);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + 
                                std::string(cudaGetErrorString(error)));
    }
}

/**
 * @brief CUDA core matrix multiplication: C = A * B
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param config CUDA core configuration
 */
template<typename T>
void cuda_core_matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C,
                      const CudaCoreConfig& config) {
    cuda_core_gemm(A, B, C, T(1), T(0), config);
}

/**
 * @brief Check if CUDA core can be used for given tensors
 * @param A Input matrix A
 * @param B Input matrix B
 * @return true if CUDA core can be used
 */
template<typename T>
bool can_use_cuda_core(const Tensor<T>& A, const Tensor<T>& B) {
    // CUDA core can handle any data type and dimensions
    // Just check if tensors are on GPU
    return A.memory_type() == MemoryType::DEVICE && 
           B.memory_type() == MemoryType::DEVICE;
}

/**
 * @brief Get optimal CUDA core configuration for tensors
 * @param A Input matrix A
 * @param B Input matrix B
 * @return Optimal CUDA core configuration
 */
template<typename T>
CudaCoreConfig get_optimal_cuda_core_config(const Tensor<T>& A, const Tensor<T>& B) {
    CudaCoreConfig config;
    
    // Get tensor dimensions
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    
    dim_t M = A_shape[A_shape.ndims() - 2];
    dim_t K = A_shape[A_shape.ndims() - 1];
    dim_t N = B_shape[B_shape.ndims() - 1];
    
    // Optimize tile sizes based on matrix dimensions
    if (M >= 64 && N >= 64) {
        config.tile_m = 32;
        config.tile_n = 32;
        config.tile_k = 32;
        config.block_m = 16;
        config.block_n = 16;
    } else {
        config.tile_m = 16;
        config.tile_n = 16;
        config.tile_k = 16;
        config.block_m = 8;
        config.block_n = 8;
    }
    
    return config;
}

/**
 * @brief Validate CUDA core contract requirements
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param config CUDA core configuration
 */
template<typename T>
void validate_cuda_core_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                                const CudaCoreConfig& config) {
    // Check if tensors are on GPU
    if (A.memory_type() != MemoryType::DEVICE) {
        throw std::runtime_error("Tensor A must be on GPU for CUDA core operations");
    }
    if (B.memory_type() != MemoryType::DEVICE) {
        throw std::runtime_error("Tensor B must be on GPU for CUDA core operations");
    }
    if (C.memory_type() != MemoryType::DEVICE) {
        throw std::runtime_error("Tensor C must be on GPU for CUDA core operations");
    }
    
    // Check tensor dimensions
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    if (A_shape.ndims() != 2 || B_shape.ndims() != 2 || C_shape.ndims() != 2) {
        throw std::runtime_error("CUDA core operations only support 2D matrices");
    }
    
    // Check matrix multiplication compatibility
    dim_t M = A_shape[0];
    dim_t K = A_shape[1];
    dim_t K2 = B_shape[0];
    dim_t N = B_shape[1];
    
    if (K != K2) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    if (C_shape[0] != M || C_shape[1] != N) {
        throw std::runtime_error("Output tensor dimensions do not match expected result");
    }
    
    // Validate configuration
    if (config.block_m <= 0 || config.block_n <= 0) {
        throw std::runtime_error("Invalid block dimensions in CUDA core configuration");
    }
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_CUDA_CORE_HPP 