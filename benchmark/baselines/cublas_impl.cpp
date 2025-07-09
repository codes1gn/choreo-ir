/**
 * @file cublas_impl.cpp
 * @brief cuBLAS baseline implementations for matrix operations
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "choreo-ir/compute/matmul.hpp"
#include "choreo-ir/utils/cuda_utils.hpp"
#include "choreo-ir/utils/debug.hpp"

namespace choreo_ir {
namespace baselines {

/**
 * @brief cuBLAS handle manager
 */
class CublasHandle {
public:
    static CublasHandle& instance() {
        static CublasHandle inst;
        return inst;
    }
    
    cublasHandle_t get() const { return handle_; }
    
private:
    cublasHandle_t handle_;
    
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    
    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }
    
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

/**
 * @brief cuBLAS matrix multiplication for float
 */
void cublas_sgemm(const float* A, const float* B, float* C,
                  index_t M, index_t N, index_t K,
                  index_t lda, index_t ldb, index_t ldc,
                  float alpha, float beta,
                  bool transpose_a = false, bool transpose_b = false,
                  cudaStream_t stream = 0) {
    auto handle = CublasHandle::instance().get();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // cuBLAS uses column-major, so we need to swap A and B
    CUBLAS_CHECK(cublasSgemm(handle, transb, transa,
                            N, M, K,
                            &alpha,
                            B, ldb,
                            A, lda,
                            &beta,
                            C, ldc));
}

/**
 * @brief cuBLAS matrix multiplication for half precision
 */
void cublas_hgemm(const __half* A, const __half* B, __half* C,
                  index_t M, index_t N, index_t K,
                  index_t lda, index_t ldb, index_t ldc,
                  float alpha, float beta,
                  bool transpose_a = false, bool transpose_b = false,
                  cudaStream_t stream = 0) {
    auto handle = CublasHandle::instance().get();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(beta);
    
    // cuBLAS uses column-major, so we need to swap A and B
    CUBLAS_CHECK(cublasHgemm(handle, transb, transa,
                            N, M, K,
                            &h_alpha,
                            B, ldb,
                            A, lda,
                            &h_beta,
                            C, ldc));
}

/**
 * @brief cuBLAS strided batched matrix multiplication for float
 */
void cublas_sgemm_strided_batched(const float* A, const float* B, float* C,
                                  index_t M, index_t N, index_t K,
                                  index_t lda, index_t ldb, index_t ldc,
                                  index_t stride_a, index_t stride_b, index_t stride_c,
                                  index_t batch_count,
                                  float alpha, float beta,
                                  bool transpose_a = false, bool transpose_b = false,
                                  cudaStream_t stream = 0) {
    auto handle = CublasHandle::instance().get();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // cuBLAS uses column-major, so we need to swap A and B
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, transb, transa,
                                          N, M, K,
                                          &alpha,
                                          B, ldb, stride_b,
                                          A, lda, stride_a,
                                          &beta,
                                          C, ldc, stride_c,
                                          batch_count));
}

/**
 * @brief cuBLAS implementation wrapper for choreo-ir interface
 */
template<typename T>
void cublas_matmul_impl(const TensorView<T>& A, const TensorView<T>& B, 
                       TensorView<T>& C, const compute::MatmulConfig& config,
                       cudaStream_t stream) {
    // Validate inputs
    CHOREO_IR_ASSERT(A.ndims() == 2, "Matrix A must be 2D");
    CHOREO_IR_ASSERT(B.ndims() == 2, "Matrix B must be 2D");
    CHOREO_IR_ASSERT(C.ndims() == 2, "Matrix C must be 2D");
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    index_t M = config.transpose_a ? A_shape[1] : A_shape[0];
    index_t K = config.transpose_a ? A_shape[0] : A_shape[1];
    index_t N = config.transpose_b ? B_shape[0] : B_shape[1];
    
    CHOREO_IR_ASSERT(C_shape[0] == M && C_shape[1] == N, "Invalid output dimensions");
    
    index_t lda = A.stride()[0];
    index_t ldb = B.stride()[0];
    index_t ldc = C.stride()[0];
    
    if constexpr (std::is_same_v<T, float>) {
        cublas_sgemm(A.data(), B.data(), C.data(),
                     M, N, K, lda, ldb, ldc,
                     config.alpha, config.beta,
                     config.transpose_a, config.transpose_b, stream);
    } else if constexpr (std::is_same_v<T, __half>) {
        cublas_hgemm(A.data(), B.data(), C.data(),
                     M, N, K, lda, ldb, ldc,
                     config.alpha, config.beta,
                     config.transpose_a, config.transpose_b, stream);
    } else {
        CHOREO_IR_ASSERT(false, "Unsupported data type for cuBLAS");
    }
}

/**
 * @brief cuBLAS batched matrix multiplication wrapper
 */
template<typename T>
void cublas_batched_matmul_impl(const TensorView<T>& A, const TensorView<T>& B,
                               TensorView<T>& C, const compute::MatmulConfig& config,
                               cudaStream_t stream) {
    // Validate inputs
    CHOREO_IR_ASSERT(A.ndims() == 3, "Batched matrix A must be 3D");
    CHOREO_IR_ASSERT(B.ndims() == 3, "Batched matrix B must be 3D");
    CHOREO_IR_ASSERT(C.ndims() == 3, "Batched matrix C must be 3D");
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    index_t batch_size = A_shape[0];
    CHOREO_IR_ASSERT(B_shape[0] == batch_size && C_shape[0] == batch_size, 
                    "Batch sizes must match");
    
    index_t M = config.transpose_a ? A_shape[2] : A_shape[1];
    index_t K = config.transpose_a ? A_shape[1] : A_shape[2];
    index_t N = config.transpose_b ? B_shape[1] : B_shape[2];
    
    index_t lda = A.stride()[1];
    index_t ldb = B.stride()[1];
    index_t ldc = C.stride()[1];
    
    index_t stride_a = A.stride()[0];
    index_t stride_b = B.stride()[0];
    index_t stride_c = C.stride()[0];
    
    if constexpr (std::is_same_v<T, float>) {
        cublas_sgemm_strided_batched(A.data(), B.data(), C.data(),
                                    M, N, K, lda, ldb, ldc,
                                    stride_a, stride_b, stride_c,
                                    batch_size,
                                    config.alpha, config.beta,
                                    config.transpose_a, config.transpose_b, stream);
    } else {
        // For other types, fall back to individual matrix multiplications
        for (index_t i = 0; i < batch_size; ++i) {
            auto A_slice = A.slice(i, i + 1);
            auto B_slice = B.slice(i, i + 1);
            auto C_slice = C.slice(i, i + 1);
            cublas_matmul_impl(A_slice, B_slice, C_slice, config, stream);
        }
    }
}

// Explicit template instantiations
template void cublas_matmul_impl<float>(const TensorView<float>&, const TensorView<float>&,
                                       TensorView<float>&, const compute::MatmulConfig&, cudaStream_t);

template void cublas_matmul_impl<__half>(const TensorView<__half>&, const TensorView<__half>&,
                                        TensorView<__half>&, const compute::MatmulConfig&, cudaStream_t);

template void cublas_batched_matmul_impl<float>(const TensorView<float>&, const TensorView<float>&,
                                               TensorView<float>&, const compute::MatmulConfig&, cudaStream_t);

} // namespace baselines
} // namespace choreo_ir 