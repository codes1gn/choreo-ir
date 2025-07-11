/**
 * @file baselines.hpp
 * @brief Baseline implementations for performance comparison
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <string>

namespace choreo_ir {
namespace baselines {

// Data layout enum
enum class DataLayout {
    ROW_MAJOR,
    COLUMN_MAJOR
};

// Base baseline interface
class Baseline {
public:
    virtual ~Baseline() = default;
    virtual void initialize() = 0;
    virtual void cleanup() = 0;
    virtual std::string name() const = 0;
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K);
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K,
               int lda, int ldb, int ldc);
    
    template<typename T>
    void axpy(int n, const T* alpha, const T* x, T* y);
    
    template<typename T>
    void copy(int n, const T* x, T* y);
};

// cuBLAS baseline implementation
class CublasBaseline : public Baseline {
public:
    explicit CublasBaseline(DataLayout layout = DataLayout::COLUMN_MAJOR);
    ~CublasBaseline() override;
    
    void initialize() override;
    void cleanup() override;
    std::string name() const override;
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K);
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K,
               int lda, int ldb, int ldc);
    
    template<typename T>
    void axpy(int n, const T* alpha, const T* x, T* y);
    
    template<typename T>
    void copy(int n, const T* x, T* y);

private:
    cublasHandle_t handle_;
    bool initialized_;
    DataLayout layout_;
};

// CUDA baseline implementation
class CudaBaseline : public Baseline {
public:
    explicit CudaBaseline(DataLayout layout = DataLayout::ROW_MAJOR);
    ~CudaBaseline() override;
    
    void initialize() override;
    void cleanup() override;
    std::string name() const override;
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K);
    
    template<typename T>
    void gemm(const T* A, const T* B, T* C, int M, int N, int K,
               int lda, int ldb, int ldc);
    
    template<typename T>
    void axpy(int n, const T* alpha, const T* x, T* y);
    
    template<typename T>
    void copy(int n, const T* x, T* y);
private:
    DataLayout layout_;
};

} // namespace baselines
} // namespace choreo_ir 