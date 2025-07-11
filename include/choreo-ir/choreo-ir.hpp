/**
 * @file choreo-ir.hpp
 * @brief Main header for Choreo-IR library
 */

#ifndef CHOREO_IR_HPP
#define CHOREO_IR_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "tensor/tensor.hpp"
#include "core/types.hpp"
#include "core/config.hpp"
#include "core/device.hpp"
#include "utils/cuda_utils.hpp"
#include "compute/validation.hpp"
#include "compute/compute.hpp"
#include "compute/wmma.hpp"
#include "compute/mma.hpp"
#include "compute/cuda_core.hpp"
#include "compute/compute_impl.hpp"


/**
 * @namespace choreo_ir
 * @brief Main namespace for the Choreo-IR library
 */
namespace choreo_ir {



// Type aliases for convenience
using TensorF32 = Tensor<float32_t>;
using TensorF16 = Tensor<float16_t>;
using TensorBF16 = Tensor<bfloat16_t>;
using TensorI32 = Tensor<int32_t>;
using TensorI8 = Tensor<int8_t>;

using TensorViewF32 = TensorView<float32_t>;
using TensorViewF16 = TensorView<float16_t>;
using TensorViewBF16 = TensorView<bfloat16_t>;
using TensorViewI32 = TensorView<int32_t>;
using TensorViewI8 = TensorView<int8_t>;

/**
 * @brief Matrix multiplication operators for convenience
 */
template<typename T>
Tensor<T> operator*(const Tensor<T>& A, const Tensor<T>& B) {
    // Validate matrix multiplication contract
    Tensor<T> C(Shape({A.shape()[A.shape().ndims() - 2], B.shape()[B.shape().ndims() - 1]}), A.memory_type());
    compute::validate_matmul_contract(A, B, C, "Matrix multiplication");
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    
    index_t M = A_shape[A_shape.ndims() - 2];
    index_t K = A_shape[A_shape.ndims() - 1];
    index_t K2 = B_shape[B_shape.ndims() - 2];
    index_t N = B_shape[B_shape.ndims() - 1];
    
    // Use compute module for device tensors, CPU for host tensors
    if (A.memory_type() == MemoryType::DEVICE) {
        // Use compute module for GPU operations
        compute::matmul(A, B, C);
    } else {
        // CPU implementation
        if constexpr (std::is_same_v<T, __half>) {
            std::fill(C.data(), C.data() + C.numel(), __float2half(0.0f));
        } else {
            std::fill(C.data(), C.data() + C.numel(), T(0));
        }
        
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    if constexpr (std::is_same_v<T, __half>) {
                        // Use proper half precision operations
                        __half a_val = A.data()[i * K + k];
                        __half b_val = B.data()[k * N + j];
                        __half c_val = C.data()[i * N + j];
                        
                        // Convert to float for computation, then back to half
                        float a_float = __half2float(a_val);
                        float b_float = __half2float(b_val);
                        float c_float = __half2float(c_val);
                        float result = c_float + a_float * b_float;
                        
                        C.data()[i * N + j] = __float2half(result);
                    } else {
                        C.data()[i * N + j] += A.data()[i * K + k] * B.data()[k * N + j];
                    }
                }
            }
        }
    }
    
    return C;
}

/**
 * @brief Mixed precision matrix multiplication
 */
template<typename T1, typename T2>
Tensor<T2> operator*(const Tensor<T1>& A, const Tensor<T2>& B) {
    // Convert A to T2 type for mixed precision
    Tensor<T2> A_converted(A.shape(), A.memory_type());
    
    if (A.memory_type() == MemoryType::DEVICE) {
        // For device tensors, we need to copy and convert
        auto host_A = Tensor<T1>(A.shape(), MemoryType::HOST);
        A.copy_to_host(host_A.data());
        
        // Convert on host
        for (size_t i = 0; i < A.numel(); ++i) {
            if constexpr (std::is_same_v<T1, __half> && std::is_same_v<T2, float>) {
                A_converted.data()[i] = __half2float(host_A.data()[i]);
            } else if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, __half>) {
                A_converted.data()[i] = __float2half(host_A.data()[i]);
            } else {
                A_converted.data()[i] = static_cast<T2>(host_A.data()[i]);
            }
        }
        
        A_converted.copy_from_host(A_converted.data());
    } else {
        // Convert on host
        for (size_t i = 0; i < A.numel(); ++i) {
            if constexpr (std::is_same_v<T1, __half> && std::is_same_v<T2, float>) {
                A_converted.data()[i] = __half2float(A.data()[i]);
            } else if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, __half>) {
                A_converted.data()[i] = __float2half(A.data()[i]);
            } else {
                A_converted.data()[i] = static_cast<T2>(A.data()[i]);
            }
        }
    }
    
    return A_converted * B;
}

/**
 * @brief Addition operators for convenience
 */
template<typename T>
Tensor<T> operator+(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.shape() != B.shape()) {
        throw std::runtime_error("Tensor shape mismatch for addition");
    }
    
    Tensor<T> C(A.shape());
    
    if (A.memory_type() == MemoryType::DEVICE) {
        // Use compute module for device addition
        // For now, fall back to CPU implementation for simplicity
        auto host_A = Tensor<T>(A.shape(), MemoryType::HOST);
        auto host_B = Tensor<T>(B.shape(), MemoryType::HOST);
        A.copy_to_host(host_A.data());
        B.copy_to_host(host_B.data());
        
        for (size_t i = 0; i < A.numel(); ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                float a_float = __half2float(host_A.data()[i]);
                float b_float = __half2float(host_B.data()[i]);
                host_A.data()[i] = __float2half(a_float + b_float);
            } else {
                host_A.data()[i] += host_B.data()[i];
            }
        }
        
        C.copy_from_host(host_A.data());
    } else {
        // CPU implementation
        for (size_t i = 0; i < A.numel(); ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                float a_float = __half2float(A.data()[i]);
                float b_float = __half2float(B.data()[i]);
                C.data()[i] = __float2half(a_float + b_float);
            } else {
                C.data()[i] = A.data()[i] + B.data()[i];
            }
        }
    }
    
    return C;
}

/**
 * @brief ReLU activation function
 */
template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> output(input.shape(), input.memory_type());
    
    if (input.memory_type() == MemoryType::DEVICE) {
        // For device tensors, convert to host for now (simplified)
        auto host_input = Tensor<T>(input.shape(), MemoryType::HOST);
        input.copy_to_host(host_input.data());
        
        auto host_output = relu(host_input);
        output.copy_from_host(host_output.data());
    } else {
        // CPU implementation
        for (size_t i = 0; i < input.numel(); ++i) {
            if constexpr (std::is_same_v<T, __half>) {
                // Convert to float for comparison, then back to half
                float input_float = __half2float(input.data()[i]);
                float result = std::max(input_float, 0.0f);
                output.data()[i] = __float2half(result);
            } else {
                output.data()[i] = std::max(input.data()[i], T(0));
            }
        }
    }
    
    return output;
}

/**
 * @brief Initialize the Choreo-IR library
 * @return true if initialization successful, false otherwise
 */
inline bool initialize() {
    return device::initialize();
}

/**
 * @brief Finalize the Choreo-IR library
 */
inline void finalize() {
    // Synchronize all CUDA operations
    cudaDeviceSynchronize();
    
    // Destroy cuBLAS handle
    auto& handle = detail::get_cublas_handle();
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
    
    // Call device cleanup
    device::finalize();
    
    // Reset device state
    cudaDeviceReset();
}

} // namespace choreo_ir

#endif // CHOREO_IR_HPP 