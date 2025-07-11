/**
 * @file cublas_baseline.hpp
 * @brief cuBLAS baseline implementation with layout support
 */

#pragma once

#include "baselines.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// Define CUBLAS_CHECK macro if not already defined
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)
#endif

namespace choreo_ir {
namespace baselines {

inline CublasBaseline::CublasBaseline(DataLayout layout) 
    : handle_(nullptr), initialized_(false), layout_(layout) {}

inline CublasBaseline::~CublasBaseline() {
    cleanup();
}

inline void CublasBaseline::initialize() {
    if (!initialized_) {
        CUBLAS_CHECK(cublasCreate(&handle_));
        initialized_ = true;
    }
}

inline void CublasBaseline::cleanup() {
    if (initialized_ && handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
        initialized_ = false;
    }
}

inline std::string CublasBaseline::name() const {
    return layout_ == DataLayout::ROW_MAJOR ? "cuBLAS-RowMajor" : "cuBLAS-ColMajor";
}

// Template specializations for different data types
template<>
inline void CublasBaseline::gemm<float>(const float* A, const float* B, float* C, 
                                        int M, int N, int K) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices
        // For row-major A(M×K) and B(K×N), we want C = A×B (row-major memory)
        // cuBLAS expects column-major, so we use CUBLAS_OP_T for both, and set lda/ldb/ldc as in row-major
        CUBLAS_CHECK(cublasSgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both matrices
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, K,                       // A matrix, lda = K
                                 B, N,                       // B matrix, ldb = N
                                 &beta,
                                 C, M));                     // C matrix, ldc = M
    } else {
        // Column-major layout: direct cuBLAS call
        // For column-major input, cuBLAS expects column-major format
        CUBLAS_CHECK(cublasSgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, M,                       // A matrix, lda = M (column-major)
                                 B, K,                       // B matrix, ldb = K (column-major)
                                 &beta,
                                 C, M));                     // C matrix, ldc = M (column-major output)
    }
}

template<>
inline void CublasBaseline::gemm<double>(const double* A, const double* B, double* C, 
                                         int M, int N, int K) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices
        CUBLAS_CHECK(cublasDgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,
                                 M, N, K,
                                 &alpha,
                                 A, K,
                                 B, N,
                                 &beta,
                                 C, M));
    } else {
        // Column-major layout: direct cuBLAS call
        CUBLAS_CHECK(cublasDgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, M,                       // A matrix, lda = M
                                 B, K,                       // B matrix, ldb = K
                                 &beta,
                                 C, M));                     // C matrix, ldc = M
    }
}

template<>
inline void CublasBaseline::gemm<__half>(const __half* A, const __half* B, __half* C, 
                                         int M, int N, int K) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices
        CUBLAS_CHECK(cublasHgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,
                                 M, N, K,
                                 &alpha,
                                 A, K,
                                 B, N,
                                 &beta,
                                 C, M));
    } else {
        // Column-major layout: direct cuBLAS call
        CUBLAS_CHECK(cublasHgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, M,                       // A matrix, lda = M
                                 B, K,                       // B matrix, ldb = K
                                 &beta,
                                 C, M));                     // C matrix, ldc = M
    }
}

// Template specializations for gemm with explicit strides
template<>
inline void CublasBaseline::gemm<float>(const float* A, const float* B, float* C, 
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices (explicit strides)
        CUBLAS_CHECK(cublasSgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,
                                 M, N, K,
                                 &alpha,
                                 A, lda,
                                 B, ldb,
                                 &beta,
                                 C, ldc));
    } else {
        // Column-major layout: direct cuBLAS call
        CUBLAS_CHECK(cublasSgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, lda,                     // A matrix with explicit lda
                                 B, ldb,                     // B matrix with explicit ldb
                                 &beta,
                                 C, ldc));                   // C matrix with explicit ldc
    }
}

template<>
inline void CublasBaseline::gemm<double>(const double* A, const double* B, double* C, 
                                         int M, int N, int K,
                                         int lda, int ldb, int ldc) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const double alpha = 1.0;
    const double beta = 0.0;
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices (explicit strides)
        CUBLAS_CHECK(cublasDgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,
                                 M, N, K,
                                 &alpha,
                                 A, lda,
                                 B, ldb,
                                 &beta,
                                 C, ldc));
    } else {
        // Column-major layout: direct cuBLAS call
        CUBLAS_CHECK(cublasDgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, lda,                     // A matrix with explicit lda
                                 B, ldb,                     // B matrix with explicit ldb
                                 &beta,
                                 C, ldc));                   // C matrix with explicit ldc
    }
}

template<>
inline void CublasBaseline::gemm<__half>(const __half* A, const __half* B, __half* C, 
                                         int M, int N, int K,
                                         int lda, int ldb, int ldc) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    
    if (layout_ == DataLayout::ROW_MAJOR) {
        // Row-major layout: simulate row-major by transposing both matrices (explicit strides)
        CUBLAS_CHECK(cublasHgemm(handle_,
                                 CUBLAS_OP_T, CUBLAS_OP_T,
                                 M, N, K,
                                 &alpha,
                                 A, lda,
                                 B, ldb,
                                 &beta,
                                 C, ldc));
    } else {
        // Column-major layout: direct cuBLAS call
        CUBLAS_CHECK(cublasHgemm(handle_,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
                                 M, N, K,                    // Result dimensions M x N
                                 &alpha,
                                 A, lda,                     // A matrix with explicit lda
                                 B, ldb,                     // B matrix with explicit ldb
                                 &beta,
                                 C, ldc));                   // C matrix with explicit ldc
    }
}

// Template specializations for axpy
template<>
inline void CublasBaseline::axpy<float>(int n, const float* alpha, const float* x, float* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    CUBLAS_CHECK(cublasSaxpy(handle_, n, alpha, x, 1, y, 1));
}

template<>
inline void CublasBaseline::axpy<double>(int n, const double* alpha, const double* x, double* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    CUBLAS_CHECK(cublasDaxpy(handle_, n, alpha, x, 1, y, 1));
}

template<>
inline void CublasBaseline::axpy<__half>(int n, const __half* alpha, const __half* x, __half* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    // cuBLAS doesn't support half precision axpy, so we set output to NaN
    // This will be handled by a separate CUDA kernel in the .cu file
    fprintf(stderr, "[WARN] CublasBaseline: axpy<__half> not supported by cuBLAS, output set to NaN.\n");
    // Set output to NaN using cudaMemset
    cudaMemset(y, 0xFF, n * sizeof(__half));
}

// Template specializations for copy
template<>
inline void CublasBaseline::copy<float>(int n, const float* x, float* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    CUBLAS_CHECK(cublasScopy(handle_, n, x, 1, y, 1));
}

template<>
inline void CublasBaseline::copy<double>(int n, const double* x, double* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    CUBLAS_CHECK(cublasDcopy(handle_, n, x, 1, y, 1));
}

template<>
inline void CublasBaseline::copy<__half>(int n, const __half* x, __half* y) {
    if (!initialized_) {
        throw std::runtime_error("CublasBaseline not initialized");
    }
    // cuBLAS doesn't support half precision copy, so we set output to NaN
    fprintf(stderr, "[WARN] CublasBaseline: copy<__half> not supported by cuBLAS, output set to NaN.\n");
    // Set output to NaN using cudaMemset
    cudaMemset(y, 0xFF, n * sizeof(__half));
}

} // namespace baselines
} // namespace choreo_ir 