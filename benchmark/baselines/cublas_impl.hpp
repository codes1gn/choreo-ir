/**
 * @file cublas_impl.hpp
 * @brief cuBLAS baseline implementations for performance comparison
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <memory>
#include <chrono>

namespace choreo_ir {
namespace baselines {

/**
 * @brief cuBLAS wrapper for easy benchmarking
 */
class CublasBaseline {
public:
    CublasBaseline() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    
    ~CublasBaseline() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }
    
    // Float GEMM
    void sgemm(int M, int N, int K, 
               const float* A, const float* B, float* C,
               float alpha = 1.0f, float beta = 0.0f) {
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, A, K, &beta, C, N));
    }
    
    // Half GEMM
    void hgemm(int M, int N, int K,
               const __half* A, const __half* B, __half* C,
               __half alpha = __float2half(1.0f), __half beta = __float2half(0.0f)) {
        CUBLAS_CHECK(cublasHgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, A, K, &beta, C, N));
    }
    
    // Mixed precision GEMM (half input, float output) with tensor cores
    void gemm_ex_mixed(int M, int N, int K,
                       const __half* A, const __half* B, float* C,
                       float alpha = 1.0f, float beta = 0.0f) {
        CUBLAS_CHECK(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                B, CUDA_R_16F, N,
                                A, CUDA_R_16F, K,
                                &beta,
                                C, CUDA_R_32F, N,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    
    // Batch GEMM
    void sgemm_strided_batched(int M, int N, int K, int batch_count,
                               const float* A, const float* B, float* C,
                               float alpha = 1.0f, float beta = 0.0f) {
        long long int stride_A = M * K;
        long long int stride_B = K * N;
        long long int stride_C = M * N;
        
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                             N, M, K,
                                             &alpha,
                                             B, N, stride_B,
                                             A, K, stride_A,
                                             &beta,
                                             C, N, stride_C,
                                             batch_count));
    }
    
    void hgemm_strided_batched(int M, int N, int K, int batch_count,
                               const __half* A, const __half* B, __half* C,
                               __half alpha = __float2half(1.0f), __half beta = __float2half(0.0f)) {
        long long int stride_A = M * K;
        long long int stride_B = K * N;
        long long int stride_C = M * N;
        
        CUBLAS_CHECK(cublasHgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                             N, M, K,
                                             &alpha,
                                             B, N, stride_B,
                                             A, K, stride_A,
                                             &beta,
                                             C, N, stride_C,
                                             batch_count));
    }
    
    cublasHandle_t handle() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

/**
 * @brief RAII wrapper for device memory
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    
    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    T* get() const { return ptr_; }
    size_t size() const { return count_; }
    
    void copy_from_host(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    size_t count_;
};

/**
 * @brief Benchmark helper for timing CUDA operations
 */
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&end_));
    }
    
    ~CudaTimer() {
        if (start_) cudaEventDestroy(start_);
        if (end_) cudaEventDestroy(end_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(end_));
        CUDA_CHECK(cudaEventSynchronize(end_));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, end_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t end_ = nullptr;
};

} // namespace baselines
} // namespace choreo_ir 