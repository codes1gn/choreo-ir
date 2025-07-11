/**
 * @file cuda_baseline.hpp
 * @brief Pure CUDA baseline implementation
 */

#pragma once

#include "baselines.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace choreo_ir {
namespace baselines {

inline CudaBaseline::CudaBaseline(DataLayout layout) : layout_(layout) {}

inline CudaBaseline::~CudaBaseline() {
    cleanup();
}

inline void CudaBaseline::initialize() {
    // Pure CUDA doesn't need special initialization
}

inline void CudaBaseline::cleanup() {
    // Pure CUDA doesn't need special cleanup
}

inline std::string CudaBaseline::name() const {
    return layout_ == DataLayout::ROW_MAJOR ? "CUDA-RowMajor" : "CUDA-ColMajor";
}

// Forward declarations for CUDA kernels
template<typename T>
void gemm_kernel_impl(const T* A, const T* B, T* C, 
                     int M, int N, int K, dim3 gridDim, dim3 blockDim, DataLayout layout);

template<typename T>
void axpy_kernel_impl(int n, const T* alpha, const T* x, T* y, 
                     int gridSize, int blockSize);

template<typename T>
void copy_kernel_impl(int n, const T* x, T* y, 
                     int gridSize, int blockSize);

template<typename T>
inline void CudaBaseline::gemm(const T* A, const T* B, T* C, 
                        int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                  (M + blockDim.y - 1) / blockDim.y);
    
    gemm_kernel_impl<T>(A, B, C, M, N, K, gridDim, blockDim, layout_);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA gemm kernel failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
}

template<typename T>
inline void CudaBaseline::gemm(const T* A, const T* B, T* C, 
                        int M, int N, int K,
                        int lda, int ldb, int ldc) {
    // For simplicity, use the same implementation as the non-strided version
    // In a real implementation, you might want to handle strides differently
    gemm(A, B, C, M, N, K);
}

template<typename T>
inline void CudaBaseline::axpy(int n, const T* alpha, const T* x, T* y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    axpy_kernel_impl<T>(n, alpha, x, y, gridSize, blockSize);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA axpy kernel failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
}

template<typename T>
inline void CudaBaseline::copy(int n, const T* x, T* y) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    copy_kernel_impl<T>(n, x, y, gridSize, blockSize);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA copy kernel failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
}

} // namespace baselines
} // namespace choreo_ir 