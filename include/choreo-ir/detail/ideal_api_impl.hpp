/**
 * @file ideal_api_impl.hpp
 * @brief Implementation details for the ideal API
 */

#pragma once

#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace choreo_ir {
namespace detail {

// Global cuBLAS handle (lazy initialization)
static cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

/**
 * @brief Fill array with random values
 */
template<typename T>
void random_fill(T* data, size_t size, T min_val, T max_val) {
    static std::random_device rd;
    static std::mt19937 gen(42); // Fixed seed for reproducibility
    
    if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < size; ++i) {
            data[i] = __float2half(dis(gen));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }
}

/**
 * @brief Copy data from host to device
 */
template<typename T>
void host_to_device_copy(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}

/**
 * @brief Copy data from device to host
 */
template<typename T>
void device_to_host_copy(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

/**
 * @brief Host matrix multiplication (CPU implementation)
 */
template<typename T>
void host_matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    // Simple CPU implementation for testing
    std::fill(C, C + M * N, T(0));
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/**
 * @brief Device matrix multiplication using cuBLAS
 */
template<typename T>
void device_matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    auto handle = get_cublas_handle();
    
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, A, K, &beta, C, N));
    } else if constexpr (std::is_same_v<T, double>) {
        const double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, A, K, &beta, C, N));
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               B, N, A, K, &beta, C, N));
    }
}

} // namespace detail

// Mixed precision matmul specializations
template<>
inline void matmul(const host_tensor<__half>& A, const host_tensor<__half>& B, host_tensor<float>& C) {
    // Convert to device, compute, then copy back
    auto d_A = A.to_device();
    auto d_B = B.to_device();
    auto d_C = device_tensor<float>::zeros(C.shape_);
    
    auto handle = detail::get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    // Use cublasGemmEx for mixed precision
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B.data(), CUDA_R_16F, N,
                            d_A.data(), CUDA_R_16F, K,
                            &beta,
                            d_C.data(), CUDA_R_32F, N,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    C = d_C.to_host();
}

template<>
inline void matmul(const device_tensor<__half>& A, const device_tensor<__half>& B, device_tensor<float>& C) {
    auto handle = detail::get_cublas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    // Use cublasGemmEx for mixed precision with tensor cores
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            B.data(), CUDA_R_16F, N,
                            A.data(), CUDA_R_16F, K,
                            &beta,
                            C.data(), CUDA_R_32F, N,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Implement missing device tensor methods
template<typename T>
device_tensor<T> device_tensor<T>::ones(std::initializer_list<int> shape) {
    device_tensor<T> tensor(shape);
    
    // Fill with ones on device
    if constexpr (std::is_same_v<T, float>) {
        auto ones_host = std::vector<float>(tensor.numel(), 1.0f);
        CUDA_CHECK(cudaMemcpy(tensor.data(), ones_host.data(), 
                            tensor.numel() * sizeof(float), cudaMemcpyHostToDevice));
    } else if constexpr (std::is_same_v<T, __half>) {
        auto ones_host = std::vector<__half>(tensor.numel(), __float2half(1.0f));
        CUDA_CHECK(cudaMemcpy(tensor.data(), ones_host.data(), 
                            tensor.numel() * sizeof(__half), cudaMemcpyHostToDevice));
    }
    
    return tensor;
}

template<typename T>
device_tensor<T> device_tensor<T>::random(std::initializer_list<int> shape, T min_val, T max_val) {
    // Generate on host then copy to device
    auto host_data = std::vector<T>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    detail::random_fill(host_data.data(), host_data.size(), min_val, max_val);
    
    device_tensor<T> tensor(shape);
    CUDA_CHECK(cudaMemcpy(tensor.data(), host_data.data(), 
                        host_data.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    return tensor;
}

template<typename T>
int device_tensor<T>::ndims() const {
    return static_cast<int>(shape_.size());
}

template<typename T>
device_tensor<T> device_tensor<T>::slice(int start, int end) const {
    // Similar to host implementation but for device
    if (shape_.empty()) return device_tensor<T>();
    
    int batch_size = end - start;
    std::vector<int> new_shape = shape_;
    new_shape[0] = batch_size;
    
    device_tensor<T> result(new_shape);
    
    size_t elements_per_batch = numel() / shape_[0];
    T* src_ptr = data_.get() + start * elements_per_batch;
    CUDA_CHECK(cudaMemcpy(result.data(), src_ptr, 
                        batch_size * elements_per_batch * sizeof(T), 
                        cudaMemcpyDeviceToDevice));
    
    return result;
}

template<typename T>
device_tensor<T> device_tensor<T>::squeeze(int dim) const {
    std::vector<int> new_shape;
    for (int i = 0; i < shape_.size(); ++i) {
        if (i != dim || shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
        }
    }
    
    device_tensor<T> result(new_shape);
    CUDA_CHECK(cudaMemcpy(result.data(), data_.get(), 
                        numel() * sizeof(T), cudaMemcpyDeviceToDevice));
    return result;
}

template<typename T>
device_tensor<T>& device_tensor<T>::operator=(const device_tensor<T>& other) {
    shape_ = other.shape_;
    allocate();
    if (data_ && other.data_) {
        CUDA_CHECK(cudaMemcpy(data_.get(), other.data_.get(), 
                            other.numel() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    return *this;
}

template<typename T>
device_tensor<T>& device_tensor<T>::operator=(const host_tensor<T>& other) {
    *this = other.to_device();
    return *this;
}

// Addition operators
template<typename T>
host_tensor<T> operator+(const host_tensor<T>& A, const host_tensor<T>& B) {
    if (A.shape() != B.shape()) {
        throw std::runtime_error("Tensor shape mismatch for addition");
    }
    
    host_tensor<T> C(A.shape_);
    for (size_t i = 0; i < A.numel(); ++i) {
        C.data()[i] = A.data()[i] + B.data()[i];
    }
    return C;
}

template<typename T>
device_tensor<T> operator+(const device_tensor<T>& A, const device_tensor<T>& B) {
    if (A.shape() != B.shape()) {
        throw std::runtime_error("Tensor shape mismatch for addition");
    }
    
    device_tensor<T> C(A.shape_);
    
    // Use cuBLAS for device addition
    auto handle = detail::get_cublas_handle();
    
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f;
        CUBLAS_CHECK(cublasScopy(handle, C.numel(), A.data(), 1, C.data(), 1));
        CUBLAS_CHECK(cublasSaxpy(handle, C.numel(), &alpha, B.data(), 1, C.data(), 1));
    } else if constexpr (std::is_same_v<T, double>) {
        const double alpha = 1.0;
        CUBLAS_CHECK(cublasDcopy(handle, C.numel(), A.data(), 1, C.data(), 1));
        CUBLAS_CHECK(cublasDaxpy(handle, C.numel(), &alpha, B.data(), 1, C.data(), 1));
    }
    
    return C;
}

// Batch matmul for device tensors
template<typename T>
device_tensor<T> batch_matmul(const device_tensor<T>& A, const device_tensor<T>& B) {
    if (A.shape_.size() < 3 || B.shape_.size() < 3) {
        throw std::runtime_error("Batch matmul requires 3D tensors");
    }
    
    int batch_size = A.shape_[0];
    int M = A.shape_[1];
    int K = A.shape_[2];
    int N = B.shape_[2];
    
    device_tensor<T> C({batch_size, M, N});
    
    auto handle = detail::get_cublas_handle();
    
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             N, M, K,
                                             &alpha,
                                             B.data(), N, K * N,
                                             A.data(), K, M * K,
                                             &beta,
                                             C.data(), N, M * N,
                                             batch_size));
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             N, M, K,
                                             &alpha,
                                             B.data(), N, K * N,
                                             A.data(), K, M * K,
                                             &beta,
                                             C.data(), N, M * N,
                                             batch_size));
    }
    
    return C;
}

// Convolution implementations
template<typename T>
device_tensor<T> conv2d(const device_tensor<T>& input, const device_tensor<T>& weight,
                        int stride, int padding) {
    // Simplified implementation - in reality would use cuDNN
    auto [N, C, H, W] = std::make_tuple(input.shape_[0], input.shape_[1], input.shape_[2], input.shape_[3]);
    auto [K, C2, R, S] = std::make_tuple(weight.shape_[0], weight.shape_[1], weight.shape_[2], weight.shape_[3]);
    
    int H_out = (H + 2 * padding - R) / stride + 1;
    int W_out = (W + 2 * padding - S) / stride + 1;
    
    return device_tensor<T>::zeros({N, K, H_out, W_out});
}

// Element-wise operations
template<typename T>
host_tensor<T> relu(const host_tensor<T>& input) {
    host_tensor<T> output(input.shape_);
    for (size_t i = 0; i < input.numel(); ++i) {
        output.data()[i] = std::max(input.data()[i], T(0));
    }
    return output;
}

// Add missing reshape method
template<typename T>
host_tensor<T> host_tensor<T>::reshape(std::initializer_list<int> new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != numel()) {
        throw std::runtime_error("Reshape size mismatch");
    }
    
    host_tensor<T> result(new_shape);
    std::copy(data_.get(), data_.get() + numel(), result.data());
    return result;
}

template<typename T>
device_tensor<T> device_tensor<T>::reshape(std::initializer_list<int> new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != numel()) {
        throw std::runtime_error("Reshape size mismatch");
    }
    
    device_tensor<T> result(new_shape);
    CUDA_CHECK(cudaMemcpy(result.data(), data_.get(), 
                        numel() * sizeof(T), cudaMemcpyDeviceToDevice));
    return result;
}

} // namespace choreo_ir 