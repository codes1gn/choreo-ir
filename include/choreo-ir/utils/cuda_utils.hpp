/**
 * @file cuda_utils.hpp
 * @brief CUDA utility functions and error handling
 */

#ifndef CHOREO_IR_UTILS_CUDA_UTILS_HPP
#define CHOREO_IR_UTILS_CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
#include <iostream>
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {
namespace cuda_utils {

/**
 * @brief Check CUDA error and throw exception if error occurs
 * @param error CUDA error code
 * @param file Source file name
 * @param line Source line number
 */
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string error_msg = "CUDA error at " + std::string(file) + ":" + std::to_string(line) + 
                               " - " + cudaGetErrorString(error);
        throw std::runtime_error(error_msg);
    }
}

/**
 * @brief Check cuBLAS error and throw exception if error occurs
 * @param error cuBLAS error code
 * @param file Source file name
 * @param line Source line number
 */
inline void check_cublas_error(cublasStatus_t error, const char* file, int line) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        std::string error_msg = "cuBLAS error at " + std::string(file) + ":" + std::to_string(line) + 
                               " - Error code: " + std::to_string(error);
        throw std::runtime_error(error_msg);
    }
}

/**
 * @brief Check cuDNN error and throw exception if error occurs
 * @param error cuDNN error code
 * @param file Source file name
 * @param line Source line number
 */
inline void check_cudnn_error(cudnnStatus_t error, const char* file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        std::string error_msg = "cuDNN error at " + std::string(file) + ":" + std::to_string(line) + 
                               " - " + cudnnGetErrorString(error);
        throw std::runtime_error(error_msg);
    }
}

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        if (config::ENABLE_CUDA_ERROR_CHECKING) { \
            choreo_ir::cuda_utils::check_cuda_error(call, __FILE__, __LINE__); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        if (config::ENABLE_CUDA_ERROR_CHECKING) { \
            choreo_ir::cuda_utils::check_cublas_error(call, __FILE__, __LINE__); \
        } \
    } while (0)

#define CUDNN_CHECK(call) \
    do { \
        if (config::ENABLE_CUDA_ERROR_CHECKING) { \
            choreo_ir::cuda_utils::check_cudnn_error(call, __FILE__, __LINE__); \
        } \
    } while (0)

/**
 * @brief Get current CUDA device ID
 * @return Device ID
 */
inline int get_current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/**
 * @brief Set CUDA device
 * @param device_id Device ID
 */
inline void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

/**
 * @brief Synchronize current device
 */
inline void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Get device properties
 * @param device_id Device ID
 * @return Device properties
 */
inline cudaDeviceProp get_device_properties(int device_id) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    return props;
}

/**
 * @brief Check if device supports tensor core
 * @param device_id Device ID
 * @return true if supported, false otherwise
 */
inline bool supports_tensor_core(int device_id) {
    cudaDeviceProp props = get_device_properties(device_id);
    return props.major >= 7;  // Tensor core available from SM 7.0+
}

/**
 * @brief Get compute capability
 * @param device_id Device ID
 * @return Compute capability
 */
inline ComputeCapability get_compute_capability(int device_id) {
    cudaDeviceProp props = get_device_properties(device_id);
    int capability = props.major * 10 + props.minor;
    return static_cast<ComputeCapability>(capability);
}

/**
 * @brief Get free and total memory
 * @param free_memory Free memory in bytes
 * @param total_memory Total memory in bytes
 */
inline void get_memory_info(size_t& free_memory, size_t& total_memory) {
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
}

/**
 * @brief Allocate device memory
 * @param size Size in bytes
 * @return Device pointer
 */
inline void* malloc_device(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

/**
 * @brief Allocate host memory
 * @param size Size in bytes
 * @return Host pointer
 */
inline void* malloc_host(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

/**
 * @brief Allocate managed memory
 * @param size Size in bytes
 * @return Managed pointer
 */
inline void* malloc_managed(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

/**
 * @brief Free device memory
 * @param ptr Device pointer
 */
inline void free_device(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

/**
 * @brief Free host memory
 * @param ptr Host pointer
 */
inline void free_host(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

/**
 * @brief Copy memory
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 * @param kind Copy kind
 */
inline void memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
}

/**
 * @brief Copy memory asynchronously
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 * @param kind Copy kind
 * @param stream CUDA stream
 */
inline void memcpy_async(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
}

/**
 * @brief Set memory
 * @param ptr Pointer
 * @param value Value
 * @param size Size in bytes
 */
inline void memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

/**
 * @brief Set memory asynchronously
 * @param ptr Pointer
 * @param value Value
 * @param size Size in bytes
 * @param stream CUDA stream
 */
inline void memset_async(void* ptr, int value, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream));
}

/**
 * @brief Create CUDA stream
 * @return CUDA stream
 */
inline cudaStream_t create_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

/**
 * @brief Destroy CUDA stream
 * @param stream CUDA stream
 */
inline void destroy_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * @brief Synchronize CUDA stream
 * @param stream CUDA stream
 */
inline void synchronize_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Create CUDA event
 * @return CUDA event
 */
inline cudaEvent_t create_event() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

/**
 * @brief Destroy CUDA event
 * @param event CUDA event
 */
inline void destroy_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventDestroy(event));
}

/**
 * @brief Record CUDA event
 * @param event CUDA event
 * @param stream CUDA stream
 */
inline void record_event(cudaEvent_t event, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(event, stream));
}

/**
 * @brief Wait for CUDA event
 * @param event CUDA event
 */
inline void wait_event(cudaEvent_t event) {
    CUDA_CHECK(cudaEventSynchronize(event));
}

/**
 * @brief Get elapsed time between events
 * @param start Start event
 * @param end End event
 * @return Elapsed time in milliseconds
 */
inline float get_elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
    return time;
}

/**
 * @brief RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    CudaStream() : stream_(create_stream()) {}
    ~CudaStream() { 
        if (stream_) destroy_stream(stream_); 
    }
    
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) destroy_stream(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() { 
        if (stream_) synchronize_stream(stream_); 
    }

private:
    cudaStream_t stream_;
};

/**
 * @brief RAII wrapper for CUDA event
 */
class CudaEvent {
public:
    CudaEvent() : event_(create_event()) {}
    ~CudaEvent() { 
        if (event_) destroy_event(event_); 
    }
    
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) destroy_event(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) { 
        if (event_) record_event(event_, stream); 
    }
    
    void wait() { 
        if (event_) wait_event(event_); 
    }
    
    float elapsed_time(const CudaEvent& other) const {
        if (event_ && other.event_) {
            return get_elapsed_time(event_, other.event_);
        }
        return 0.0f;
    }

private:
    cudaEvent_t event_;
};

} // namespace cuda_utils
} // namespace choreo_ir

#endif // CHOREO_IR_UTILS_CUDA_UTILS_HPP 