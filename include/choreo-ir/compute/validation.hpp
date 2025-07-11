/**
 * @file validation.hpp
 * @brief Contract validation for compute operations
 */

#ifndef CHOREO_IR_COMPUTE_VALIDATION_HPP
#define CHOREO_IR_COMPUTE_VALIDATION_HPP

#include <stdexcept>
#include <string>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @brief Validation error class
 */
class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Validate matrix multiplication contract
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param operation_name Operation name for error messages
 */
template<typename T>
void validate_matmul_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, 
                            const std::string& operation_name = "Matrix multiplication") {
    // Check tensor dimensions
    if (A.ndims() < 2 || B.ndims() < 2) {
        throw ValidationError(operation_name + ": Input tensors must be at least 2D");
    }
    
    if (C.ndims() < 2) {
        throw ValidationError(operation_name + ": Output tensor must be at least 2D");
    }
    
    // Get matrix dimensions
    index_t M = A.shape()[A.shape().ndims() - 2];
    index_t K = A.shape()[A.shape().ndims() - 1];
    index_t K2 = B.shape()[B.shape().ndims() - 2];
    index_t N = B.shape()[B.shape().ndims() - 1];
    index_t M_out = C.shape()[C.shape().ndims() - 2];
    index_t N_out = C.shape()[C.shape().ndims() - 1];
    
    // Check dimension compatibility
    if (K != K2) {
        throw ValidationError(operation_name + ": Matrix dimension mismatch: A(" + 
                            std::to_string(M) + "x" + std::to_string(K) + ") * B(" + 
                            std::to_string(K2) + "x" + std::to_string(N) + ")");
    }
    
    if (M != M_out || N != N_out) {
        throw ValidationError(operation_name + ": Output shape mismatch: expected (" + 
                            std::to_string(M) + "x" + std::to_string(N) + "), got (" + 
                            std::to_string(M_out) + "x" + std::to_string(N_out) + ")");
    }
    
    // Check memory type consistency
    if (A.memory_type() != B.memory_type() || A.memory_type() != C.memory_type()) {
        throw ValidationError(operation_name + ": Memory type mismatch between input and output tensors");
    }
    
    // Check data type consistency
    if (A.dtype() != B.dtype() || A.dtype() != C.dtype()) {
        throw ValidationError(operation_name + ": Data type mismatch between input and output tensors");
    }
    
    // Check for null pointers
    if (A.data() == nullptr) {
        throw ValidationError(operation_name + ": Input tensor A has null data pointer");
    }
    
    if (B.data() == nullptr) {
        throw ValidationError(operation_name + ": Input tensor B has null data pointer");
    }
    
    if (C.data() == nullptr) {
        throw ValidationError(operation_name + ": Output tensor C has null data pointer");
    }
    
    // Check for empty tensors
    if (A.numel() == 0) {
        throw ValidationError(operation_name + ": Input tensor A is empty");
    }
    
    if (B.numel() == 0) {
        throw ValidationError(operation_name + ": Input tensor B is empty");
    }
    
    if (C.numel() == 0) {
        throw ValidationError(operation_name + ": Output tensor C is empty");
    }
    
    // Check for valid dimensions (positive)
    if (M <= 0 || N <= 0 || K <= 0) {
        throw ValidationError(operation_name + ": Invalid matrix dimensions: M=" + 
                            std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K));
    }
}

/**
 * @brief Validate cuBLAS matrix multiplication contract
 * @param A Input matrix A
 * @param B Input matrix B  
 * @param C Output matrix C
 * @param transA Transpose flag for A
 * @param transB Transpose flag for B
 * @param operation_name Operation name for error messages
 */
template<typename T>
void validate_cublas_matmul_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                                    bool transA, bool transB, 
                                    const std::string& operation_name = "cuBLAS matrix multiplication") {
    // First validate basic matrix multiplication contract
    validate_matmul_contract(A, B, C, operation_name);
    
    // Get matrix dimensions
    index_t M = A.shape()[A.shape().ndims() - 2];
    index_t K = A.shape()[A.shape().ndims() - 1];
    index_t K2 = B.shape()[B.shape().ndims() - 2];
    index_t N = B.shape()[B.shape().ndims() - 1];
    
    // Calculate effective dimensions for cuBLAS
    index_t m = transA ? K : M;
    index_t k = transA ? M : K;
    index_t n = transB ? K2 : N;
    index_t k2 = transB ? N : K2;
    
    // Check cuBLAS-specific requirements
    if (k != k2) {
        throw ValidationError(operation_name + ": cuBLAS dimension mismatch after transpose: k=" + 
                            std::to_string(k) + ", k2=" + std::to_string(k2));
    }
    
    // Check leading dimensions
    index_t lda = transA ? M : K;
    index_t ldb = transB ? N : K2;
    index_t ldc = N;
    
    if (lda <= 0 || ldb <= 0 || ldc <= 0) {
        throw ValidationError(operation_name + ": Invalid leading dimensions: lda=" + 
                            std::to_string(lda) + ", ldb=" + std::to_string(ldb) + 
                            ", ldc=" + std::to_string(ldc));
    }
    
    // Check that tensors are contiguous (required for cuBLAS)
    if (!A.is_contiguous()) {
        throw ValidationError(operation_name + ": Input tensor A must be contiguous for cuBLAS");
    }
    
    if (!B.is_contiguous()) {
        throw ValidationError(operation_name + ": Input tensor B must be contiguous for cuBLAS");
    }
    
    if (!C.is_contiguous()) {
        throw ValidationError(operation_name + ": Output tensor C must be contiguous for cuBLAS");
    }
}

/**
 * @brief Validate element-wise operation contract
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param operation_name Operation name for error messages
 */
template<typename T>
void validate_elementwise_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                                 const std::string& operation_name = "Element-wise operation") {
    // Check shape compatibility
    if (A.shape() != B.shape() || A.shape() != C.shape()) {
        throw ValidationError(operation_name + ": Shape mismatch between input and output tensors");
    }
    
    // Check memory type consistency
    if (A.memory_type() != B.memory_type() || A.memory_type() != C.memory_type()) {
        throw ValidationError(operation_name + ": Memory type mismatch between input and output tensors");
    }
    
    // Check data type consistency
    if (A.dtype() != B.dtype() || A.dtype() != C.dtype()) {
        throw ValidationError(operation_name + ": Data type mismatch between input and output tensors");
    }
    
    // Check for null pointers
    if (A.data() == nullptr || B.data() == nullptr || C.data() == nullptr) {
        throw ValidationError(operation_name + ": Tensor has null data pointer");
    }
    
    // Check for empty tensors
    if (A.numel() == 0) {
        throw ValidationError(operation_name + ": Input tensors are empty");
    }
}

/**
 * @brief Validate tensor core requirements
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param operation_name Operation name for error messages
 */
template<typename T>
void validate_tensor_core_contract(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
                                  const std::string& operation_name = "Tensor core operation") {
    // Check data type support
    if constexpr (std::is_same_v<T, __half>) {
        // Half precision is supported
    } else if constexpr (std::is_same_v<T, float>) {
        // Float precision is supported
    } else if constexpr (std::is_same_v<T, double>) {
        throw ValidationError(operation_name + ": Double precision not supported for tensor core operations");
    } else {
        throw ValidationError(operation_name + ": Data type not supported for tensor core operations");
    }
    
    // Check matrix dimensions (tensor core has specific requirements)
    index_t M = A.shape()[A.shape().ndims() - 2];
    index_t K = A.shape()[A.shape().ndims() - 1];
    index_t N = B.shape()[B.shape().ndims() - 1];
    
    // Tensor core typically requires dimensions to be multiples of 8 or 16
    if (M % 8 != 0 || N % 8 != 0 || K % 8 != 0) {
        throw ValidationError(operation_name + ": Tensor core requires dimensions to be multiples of 8: " +
                            "M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K));
    }
    
    // Check alignment requirements
    if (reinterpret_cast<uintptr_t>(A.data()) % 16 != 0) {
        throw ValidationError(operation_name + ": Tensor A data not 16-byte aligned");
    }
    
    if (reinterpret_cast<uintptr_t>(B.data()) % 16 != 0) {
        throw ValidationError(operation_name + ": Tensor B data not 16-byte aligned");
    }
    
    if (reinterpret_cast<uintptr_t>(C.data()) % 16 != 0) {
        throw ValidationError(operation_name + ": Tensor C data not 16-byte aligned");
    }
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_VALIDATION_HPP 