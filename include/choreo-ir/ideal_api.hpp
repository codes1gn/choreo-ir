/**
 * @file ideal_api.hpp
 * @brief Ideal programming experience API for Choreo-IR
 * 
 * This header provides the high-level, intuitive tensor operations that make
 * CUDA programming feel like mathematical notation.
 * 
 * Example usage:
 *   auto A = host_tensor<float>::random({1024, 512});
 *   auto B = host_tensor<float>::random({512, 256}); 
 *   auto C = A * B;  // Matrix multiplication, just like math!
 */

#pragma once

#include <choreo-ir/core/types.hpp>
#include <choreo-ir/core/config.hpp>
#include <choreo-ir/tensor/tensor.hpp>
#include <choreo-ir/utils/cuda_utils.hpp>

#include <memory>
#include <tuple>
#include <vector>
#include <initializer_list>
#include <stdexcept>

namespace choreo_ir {

/**
 * @brief Forward declarations
 */
template<typename T> class host_tensor;
template<typename T> class device_tensor;

/**
 * @brief Host tensor with intuitive API
 */
template<typename T>
class host_tensor {
public:
    // Construction
    host_tensor() = default;
    host_tensor(std::initializer_list<int> shape);
    host_tensor(const std::vector<int>& shape);
    
    // Static factory methods
    static host_tensor<T> zeros(std::initializer_list<int> shape);
    static host_tensor<T> ones(std::initializer_list<int> shape);
    static host_tensor<T> random(std::initializer_list<int> shape, T min_val = T{}, T max_val = T{});
    
    // Properties
    std::tuple<int, int> shape() const;
    size_t numel() const;
    int ndims() const;
    T* data() const;
    
    // Device transfer
    device_tensor<T> to_device() const;
    bool is_on_device() const { return false; }
    
    // Element access
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    
    // Slicing and reshaping
    host_tensor<T> slice(int start, int end) const;
    host_tensor<T> squeeze(int dim) const;
    host_tensor<T> reshape(std::initializer_list<int> new_shape) const;
    
    // Assignment
    host_tensor<T>& operator=(const host_tensor<T>& other);
    host_tensor<T>& operator=(const device_tensor<T>& other);

private:
    std::vector<int> shape_;
    std::shared_ptr<T[]> data_;
    
    void allocate();
    friend device_tensor<T>;
};

/**
 * @brief Device tensor for GPU operations
 */
template<typename T>
class device_tensor {
public:
    // Construction
    device_tensor() = default;
    device_tensor(std::initializer_list<int> shape);
    device_tensor(const std::vector<int>& shape);
    
    // Static factory methods
    static device_tensor<T> zeros(std::initializer_list<int> shape);
    static device_tensor<T> ones(std::initializer_list<int> shape);
    static device_tensor<T> random(std::initializer_list<int> shape, T min_val = T{}, T max_val = T{});
    
    // Properties
    std::tuple<int, int> shape() const;
    size_t numel() const;
    int ndims() const;
    T* data() const;
    
    // Host transfer
    host_tensor<T> to_host() const;
    bool is_on_device() const { return true; }
    
    // Slicing and reshaping
    device_tensor<T> slice(int start, int end) const;
    device_tensor<T> squeeze(int dim) const;
    device_tensor<T> reshape(std::initializer_list<int> new_shape) const;
    
    // Assignment
    device_tensor<T>& operator=(const device_tensor<T>& other);
    device_tensor<T>& operator=(const host_tensor<T>& other);

private:
    std::vector<int> shape_;
    std::shared_ptr<T[]> data_;
    
    void allocate();
    friend host_tensor<T>;
};

// Matrix multiplication operators
template<typename T>
host_tensor<T> operator*(const host_tensor<T>& A, const host_tensor<T>& B);

template<typename T>
device_tensor<T> operator*(const device_tensor<T>& A, const device_tensor<T>& B);

// Addition operators  
template<typename T>
host_tensor<T> operator+(const host_tensor<T>& A, const host_tensor<T>& B);

template<typename T>
device_tensor<T> operator+(const device_tensor<T>& A, const device_tensor<T>& B);

/**
 * @brief Explicit matrix multiplication functions
 */
template<typename T>
void matmul(const host_tensor<T>& A, const host_tensor<T>& B, host_tensor<T>& C);

template<typename T>
void matmul(const device_tensor<T>& A, const device_tensor<T>& B, device_tensor<T>& C);

// Mixed precision support
template<typename InputT, typename OutputT>
void matmul(const host_tensor<InputT>& A, const host_tensor<InputT>& B, host_tensor<OutputT>& C);

template<typename InputT, typename OutputT> 
void matmul(const device_tensor<InputT>& A, const device_tensor<InputT>& B, device_tensor<OutputT>& C);

/**
 * @brief Batch operations
 */
template<typename T>
host_tensor<T> batch_matmul(const host_tensor<T>& A, const host_tensor<T>& B);

template<typename T>
device_tensor<T> batch_matmul(const device_tensor<T>& A, const device_tensor<T>& B);

/**
 * @brief Convolution operations  
 */
template<typename T>
host_tensor<T> conv2d(const host_tensor<T>& input, const host_tensor<T>& weight, 
                      int stride = 1, int padding = 0);

template<typename T>
device_tensor<T> conv2d(const device_tensor<T>& input, const device_tensor<T>& weight,
                        int stride = 1, int padding = 0);

/**
 * @brief Element-wise operations
 */
template<typename T>
host_tensor<T> relu(const host_tensor<T>& input);

template<typename T>
device_tensor<T> relu(const device_tensor<T>& input);

/**
 * @brief Implementation details (simplified stub versions)
 */
namespace detail {
    template<typename T>
    void random_fill(T* data, size_t size, T min_val, T max_val);
    
    template<typename T>
    void host_to_device_copy(T* dst, const T* src, size_t size);
    
    template<typename T>
    void device_to_host_copy(T* dst, const T* src, size_t size);
    
    template<typename T>
    void host_matmul(const T* A, const T* B, T* C, int M, int N, int K);
    
    template<typename T>
    void device_matmul(const T* A, const T* B, T* C, int M, int N, int K);
}

// Template implementations
template<typename T>
host_tensor<T>::host_tensor(std::initializer_list<int> shape) : shape_(shape) {
    allocate();
}

template<typename T>
host_tensor<T>::host_tensor(const std::vector<int>& shape) : shape_(shape) {
    allocate();
}

template<typename T>
void host_tensor<T>::allocate() {
    if (shape_.empty()) return;
    
    size_t total_size = 1;
    for (int dim : shape_) {
        total_size *= dim;
    }
    
    data_ = std::shared_ptr<T[]>(new T[total_size]);
}

template<typename T>
host_tensor<T> host_tensor<T>::zeros(std::initializer_list<int> shape) {
    host_tensor<T> tensor(shape);
    std::fill_n(tensor.data(), tensor.numel(), T(0));
    return tensor;
}

template<typename T>
host_tensor<T> host_tensor<T>::ones(std::initializer_list<int> shape) {
    host_tensor<T> tensor(shape);
    std::fill_n(tensor.data(), tensor.numel(), T(1));
    return tensor;
}

template<typename T>
host_tensor<T> host_tensor<T>::random(std::initializer_list<int> shape, T min_val, T max_val) {
    host_tensor<T> tensor(shape);
    detail::random_fill(tensor.data(), tensor.numel(), min_val, max_val);
    return tensor;
}

template<typename T>
std::tuple<int, int> host_tensor<T>::shape() const {
    if (shape_.size() >= 2) {
        return std::make_tuple(shape_[0], shape_[1]);
    } else if (shape_.size() == 1) {
        return std::make_tuple(shape_[0], 1);
    } else {
        return std::make_tuple(0, 0);
    }
}

template<typename T>
size_t host_tensor<T>::numel() const {
    size_t size = 1;
    for (int dim : shape_) {
        size *= dim;
    }
    return size;
}

template<typename T>
int host_tensor<T>::ndims() const {
    return static_cast<int>(shape_.size());
}

template<typename T>
T* host_tensor<T>::data() const {
    return data_.get();
}

template<typename T>
device_tensor<T> host_tensor<T>::to_device() const {
    device_tensor<T> dev_tensor(shape_);
    if (data_ && dev_tensor.data()) {
        detail::host_to_device_copy(dev_tensor.data(), data_.get(), numel());
    }
    return dev_tensor;
}

template<typename T>
T& host_tensor<T>::operator[](size_t index) {
    return data_[index];
}

template<typename T>
const T& host_tensor<T>::operator[](size_t index) const {
    return data_[index];
}

template<typename T>
host_tensor<T> host_tensor<T>::slice(int start, int end) const {
    // Simplified implementation - just create new tensor with slice of data
    if (shape_.empty()) return host_tensor<T>();
    
    int batch_size = end - start;
    std::vector<int> new_shape = shape_;
    new_shape[0] = batch_size;
    
    host_tensor<T> result(new_shape);
    
    size_t elements_per_batch = numel() / shape_[0];
    T* src_ptr = data_.get() + start * elements_per_batch;
    std::copy(src_ptr, src_ptr + batch_size * elements_per_batch, result.data());
    
    return result;
}

template<typename T>
host_tensor<T> host_tensor<T>::squeeze(int dim) const {
    std::vector<int> new_shape;
    for (int i = 0; i < shape_.size(); ++i) {
        if (i != dim || shape_[i] != 1) {
            new_shape.push_back(shape_[i]);
        }
    }
    
    host_tensor<T> result(new_shape);
    std::copy(data_.get(), data_.get() + numel(), result.data());
    return result;
}

template<typename T>
host_tensor<T>& host_tensor<T>::operator=(const host_tensor<T>& other) {
    shape_ = other.shape_;
    allocate();
    if (data_ && other.data_) {
        std::copy(other.data_.get(), other.data_.get() + other.numel(), data_.get());
    }
    return *this;
}

template<typename T>
host_tensor<T>& host_tensor<T>::operator=(const device_tensor<T>& other) {
    *this = other.to_host();
    return *this;
}

// Device tensor implementations (similar structure)
template<typename T>
device_tensor<T>::device_tensor(std::initializer_list<int> shape) : shape_(shape) {
    allocate();
}

template<typename T>
device_tensor<T>::device_tensor(const std::vector<int>& shape) : shape_(shape) {
    allocate();
}

template<typename T>
void device_tensor<T>::allocate() {
    if (shape_.empty()) return;
    
    size_t total_size = 1;
    for (int dim : shape_) {
        total_size *= dim;
    }
    
    T* raw_ptr;
    CUDA_CHECK(cudaMalloc(&raw_ptr, total_size * sizeof(T)));
    data_ = std::shared_ptr<T[]>(raw_ptr, [](T* p) { cudaFree(p); });
}

template<typename T>
device_tensor<T> device_tensor<T>::zeros(std::initializer_list<int> shape) {
    device_tensor<T> tensor(shape);
    CUDA_CHECK(cudaMemset(tensor.data(), 0, tensor.numel() * sizeof(T)));
    return tensor;
}

template<typename T>
std::tuple<int, int> device_tensor<T>::shape() const {
    if (shape_.size() >= 2) {
        return std::make_tuple(shape_[0], shape_[1]);
    } else if (shape_.size() == 1) {
        return std::make_tuple(shape_[0], 1);
    } else {
        return std::make_tuple(0, 0);
    }
}

template<typename T>
size_t device_tensor<T>::numel() const {
    size_t size = 1;
    for (int dim : shape_) {
        size *= dim;
    }
    return size;
}

template<typename T>
T* device_tensor<T>::data() const {
    return data_.get();
}

template<typename T>
host_tensor<T> device_tensor<T>::to_host() const {
    host_tensor<T> host_tensor(shape_);
    if (data_ && host_tensor.data()) {
        detail::device_to_host_copy(host_tensor.data(), data_.get(), numel());
    }
    return host_tensor;
}

// Operator implementations
template<typename T>
host_tensor<T> operator*(const host_tensor<T>& A, const host_tensor<T>& B) {
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    
    host_tensor<T> C = host_tensor<T>::zeros({M, N});
    detail::host_matmul(A.data(), B.data(), C.data(), M, N, K);
    return C;
}

template<typename T>
device_tensor<T> operator*(const device_tensor<T>& A, const device_tensor<T>& B) {
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    if (K != K2) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    
    device_tensor<T> C = device_tensor<T>::zeros({M, N});
    detail::device_matmul(A.data(), B.data(), C.data(), M, N, K);
    return C;
}

template<typename T>
void matmul(const host_tensor<T>& A, const host_tensor<T>& B, host_tensor<T>& C) {
    C = A * B;
}

template<typename T>
void matmul(const device_tensor<T>& A, const device_tensor<T>& B, device_tensor<T>& C) {
    C = A * B;
}

template<typename T>
host_tensor<T> batch_matmul(const host_tensor<T>& A, const host_tensor<T>& B) {
    // Simplified: assume batch is first dimension
    if (A.shape_.size() < 3 || B.shape_.size() < 3) {
        throw std::runtime_error("Batch matmul requires 3D tensors");
    }
    
    int batch_size = A.shape_[0];
    int M = A.shape_[1];
    int K = A.shape_[2];
    int N = B.shape_[2];
    
    host_tensor<T> C({batch_size, M, N});
    
    // Process each batch
    for (int b = 0; b < batch_size; ++b) {
        auto A_slice = A.slice(b, b + 1).squeeze(0);
        auto B_slice = B.slice(b, b + 1).squeeze(0);
        auto C_batch = A_slice * B_slice;
        
        // Copy result back (simplified)
        T* C_ptr = C.data() + b * M * N;
        std::copy(C_batch.data(), C_batch.data() + M * N, C_ptr);
    }
    
    return C;
}

template<typename T>
host_tensor<T> conv2d(const host_tensor<T>& input, const host_tensor<T>& weight, 
                      int stride, int padding) {
    // Simplified implementation - just return same size output
    auto [N, C, H, W] = std::make_tuple(input.shape_[0], input.shape_[1], input.shape_[2], input.shape_[3]);
    auto [K, C2, R, S] = std::make_tuple(weight.shape_[0], weight.shape_[1], weight.shape_[2], weight.shape_[3]);
    
    // Calculate output size with padding and stride
    int H_out = (H + 2 * padding - R) / stride + 1;
    int W_out = (W + 2 * padding - S) / stride + 1;
    
    return host_tensor<T>::zeros({N, K, H_out, W_out});
}

} // namespace choreo_ir

// Include implementation details
#include "detail/ideal_api_impl.hpp" 