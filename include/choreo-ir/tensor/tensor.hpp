/**
 * @file tensor.hpp
 * @brief Core tensor class with zero-cost abstractions and memory hierarchy support
 */

#ifndef CHOREO_IR_TENSOR_TENSOR_HPP
#define CHOREO_IR_TENSOR_TENSOR_HPP

#include <memory>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <cassert>
#include <cuda_runtime.h>
#include "layout.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"
#include "../core/device.hpp"

// Using C++17 std::is_same_v

namespace choreo_ir {

/**
 * @enum MemorySpace
 * @brief Memory hierarchy levels for CUDA programming
 */
enum class MemorySpace {
    GLOBAL,    // Global device memory
    SHARED,    // Block-level shared memory
    LOCAL,     // Thread-local memory (registers)
    TEXTURE,   // Texture memory (read-only)
    CONSTANT   // Constant memory (read-only)
};

/**
 * @class TensorView
 * @brief Non-owning view of tensor data with layout and memory space information
 */
template<typename T>
class TensorView {
public:
    /**
     * @brief Constructor from data pointer, layout, and memory space
     * @param data Data pointer
     * @param layout Tensor layout
     * @param memory_space Memory space where data resides
     */
    TensorView(T* data, const Layout& layout, MemorySpace memory_space = MemorySpace::GLOBAL) 
        : data_(data), layout_(layout), memory_space_(memory_space) {}

    /**
     * @brief Get data pointer
     * @return Data pointer
     */
    __device__ __host__ T* data() const { return data_; }

    /**
     * @brief Get tensor layout
     * @return Tensor layout
     */
    __device__ __host__ const Layout& layout() const { return layout_; }

    /**
     * @brief Get tensor shape
     * @return Tensor shape
     */
    __device__ __host__ const Shape& shape() const { return layout_.shape(); }

    /**
     * @brief Get tensor stride
     * @return Tensor stride
     */
    __device__ __host__ const Stride& stride() const { return layout_.stride(); }

    /**
     * @brief Get memory space
     * @return Memory space
     */
    __device__ __host__ MemorySpace memory_space() const { return memory_space_; }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    __device__ __host__ dim_t ndims() const { return layout_.ndims(); }

    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    __device__ __host__ index_t numel() const { return layout_.numel(); }

    /**
     * @brief Get data type
     * @return Data type
     */
    __device__ __host__ DataType dtype() const { return get_dtype<T>(); }

    /**
     * @brief Check if tensor is contiguous
     * @return true if contiguous, false otherwise
     */
    __device__ __host__ bool is_contiguous() const { return layout_.is_contiguous(); }

    /**
     * @brief Check if tensor is coalesced for GPU access
     * @return true if coalesced, false otherwise
     */
    __device__ __host__ bool is_coalesced() const { return layout_.is_coalesced(); }

    /**
     * @brief Access element at multi-dimensional indices
     * @param indices Multi-dimensional indices
     * @return Reference to element
     */
    template<typename... Args>
    __device__ __host__ T& operator()(Args... indices) const {
        return data_[layout_.offset(indices...)];
    }

    /**
     * @brief Create a view of a slice
     * @param start Start dimension
     * @param end End dimension (exclusive)
     * @return Sliced view
     */
    __device__ __host__ TensorView<T> slice(dim_t start, dim_t end) const {
        return TensorView<T>(data_, layout_.slice(start, end), memory_space_);
    }

    /**
     * @brief Create a transposed view
     * @return Transposed view
     */
    __device__ __host__ TensorView<T> transpose() const {
        return TensorView<T>(data_, layout_.transpose(), memory_space_);
    }

    /**
     * @brief Create a tiled view for data transfer operations
     * @param tile_shape Tile shape
     * @return Tiled view
     */
    __device__ __host__ TensorView<T> tile(const Shape& tile_shape) const {
        return TensorView<T>(data_, layout_.tile(tile_shape), memory_space_);
    }

    /**
     * @brief Create a reshaped view (requires contiguous tensor)
     * @param new_shape New shape
     * @return Reshaped view
     */
    __device__ __host__ TensorView<T> reshape(const Shape& new_shape) const {
        return TensorView<T>(data_, layout_.reshape(new_shape), memory_space_);
    }

    /**
     * @brief Copy data to shared memory (device function, device-only)
     * @param shared_ptr Shared memory pointer
     * @param tile_shape Shape of the tile to copy
     * @return TensorView pointing to shared memory
     */
    template<dim_t MAX_SHARED_SIZE = 49152> // 48KB default shared memory
    __device__ TensorView<T> copy_to_shared(T* shared_ptr, const Shape& tile_shape) const {
#if defined(__CUDA_ARCH__)
        // Device implementation
        index_t copy_size = std::min(tile_shape.numel(), numel());
        #pragma unroll
        for (index_t i = 0; i < copy_size; ++i) {
            shared_ptr[i] = data_[i];
        }
        __syncthreads();
        Layout shared_layout(tile_shape);
        return TensorView<T>(shared_ptr, shared_layout, MemorySpace::SHARED);
#else
        // Host compilation - return dummy view
        return TensorView<T>(nullptr, Layout(), MemorySpace::GLOBAL);
#endif
    }

    /**
     * @brief Copy data to local memory (registers, device-only)
     * @param local_array Local array to copy to
     * @param num_elements Number of elements to copy
     * @return TensorView pointing to local memory
     */
    template<dim_t LOCAL_SIZE>
    __device__ TensorView<T> copy_to_local(T (&local_array)[LOCAL_SIZE]) const {
#if defined(__CUDA_ARCH__)
        static_assert(LOCAL_SIZE > 0, "Local array size must be positive");
        index_t copy_size = std::min(LOCAL_SIZE, static_cast<dim_t>(numel()));
        #pragma unroll
        for (dim_t i = 0; i < copy_size; ++i) {
            local_array[i] = data_[i];
        }
        Shape local_shape({copy_size});
        Layout local_layout(local_shape);
        return TensorView<T>(local_array, local_layout, MemorySpace::LOCAL);
#else
        // Host compilation - return dummy view
        return TensorView<T>(nullptr, Layout(), MemorySpace::GLOBAL);
#endif
    }

    /**
     * @brief Asynchronous copy from global to shared memory (device-only)
     * @param shared_ptr Shared memory destination
     * @param tile_shape Shape of tile to copy
     * @param stream CUDA stream for async copy
     * @return TensorView pointing to shared memory
     */
    __device__ TensorView<T> async_copy_to_shared(T* shared_ptr, const Shape& tile_shape, 
                                                 cudaStream_t stream = 0) const {
#if defined(__CUDA_ARCH__)
        // Use cp.async instructions if available (Ampere+)
        #if __CUDA_ARCH__ >= 800
            // Implementation would use cp.async for memory-level parallelism
            return copy_to_shared(shared_ptr, tile_shape);
        #else
            return copy_to_shared(shared_ptr, tile_shape);
        #endif
#else
        // Host compilation - return dummy view
        return TensorView<T>(nullptr, Layout(), MemorySpace::GLOBAL);
#endif
    }

    /**
     * @brief Assignment operator for data transfer: dst = src.tile(shape)
     * This is the core abstraction for expressing data movement
     */
    template<typename SrcT>
    __device__ TensorView<T>& operator=(const TensorView<SrcT>& src) {
        // Type checking
        static_assert(std::is_convertible<SrcT, T>::value, "Source type must be convertible to destination type");
        
        // Memory space transfer logic
        if (memory_space_ == MemorySpace::SHARED && src.memory_space() == MemorySpace::GLOBAL) {
            // Global -> Shared transfer
            return *this = src.copy_to_shared(data_, layout_.shape());
        } else if (memory_space_ == MemorySpace::LOCAL && src.memory_space() == MemorySpace::SHARED) {
            // Shared -> Local transfer (would need proper local array handling)
            // This is conceptual - actual implementation would be more complex
        } else {
            // Direct copy for same memory space
            index_t min_elements = std::min(numel(), src.numel());
            for (index_t i = 0; i < min_elements; ++i) {
                data_[i] = static_cast<T>(src.data()[i]);
            }
        }
        
        return *this;
    }

private:
    T* data_;
    Layout layout_;
    MemorySpace memory_space_;
};

/**
 * @class LocalTensor
 * @brief Local memory tensor (registers)
 */
template<typename T, dim_t SIZE>
class LocalTensor {
public:
    __device__ LocalTensor() = default;
    
    __device__ LocalTensor(const Shape& shape) : layout_(shape) {
        static_assert(SIZE > 0, "Local tensor size must be positive");
    }
    
    __device__ TensorView<T> view() {
        return TensorView<T>(data_, layout_, MemorySpace::LOCAL);
    }
    
    __device__ const TensorView<T> view() const {
        return TensorView<T>(const_cast<T*>(data_), layout_, MemorySpace::LOCAL);
    }
    
    __device__ T& operator[](index_t i) { return data_[i]; }
    __device__ const T& operator[](index_t i) const { return data_[i]; }
    
private:
    T data_[SIZE];
    Layout layout_;
};

/**
 * @class Tensor
 * @brief Owning tensor class with automatic memory management (global memory)
 */
template<typename T>
class Tensor {
public:
    /**
     * @brief Default constructor - creates empty tensor
     */
    Tensor() : data_(nullptr), layout_(), memory_type_(MemoryType::DEVICE) {
        // printf("[Tensor] Default constructed, data_=%p\n", (void*)data_);
    }

    /**
     * @brief Constructor from shape
     * @param shape Tensor shape
     * @param memory_type Memory type (default: device)
     */
    explicit Tensor(const Shape& shape, MemoryType memory_type = MemoryType::DEVICE)
        : layout_(shape), memory_type_(memory_type) {
        // printf("[Tensor] Constructed with shape, shape.numel()=%ld\n", shape.numel());
        allocate();
        // printf("[Tensor] After allocate, data_=%p\n", (void*)data_);
    }

    /**
     * @brief Constructor from layout
     * @param layout Tensor layout
     * @param memory_type Memory type (default: device)
     */
    explicit Tensor(const Layout& layout, MemoryType memory_type = MemoryType::DEVICE)
        : layout_(layout), memory_type_(memory_type) {
        // printf("[Tensor] Constructed with layout, layout.numel()=%ld\n", layout.numel());
        allocate();
        // printf("[Tensor] After allocate, data_=%p\n", (void*)data_);
    }

    /**
     * @brief Copy constructor
     * @param other Other tensor
     */
    Tensor(const Tensor<T>& other) : layout_(other.layout_), memory_type_(other.memory_type_) {
        allocate();
        copy_from(other);
    }

    /**
     * @brief Move constructor
     * @param other Other tensor
     */
    Tensor(Tensor<T>&& other) noexcept 
        : data_(other.data_), layout_(std::move(other.layout_)), memory_type_(other.memory_type_) {
        other.data_ = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~Tensor() {
        deallocate();
    }

    /**
     * @brief Copy assignment operator
     * @param other Other tensor
     * @return Reference to this tensor
     */
    Tensor<T>& operator=(const Tensor<T>& other) {
        if (this != &other) {
            deallocate();
            layout_ = other.layout_;
            memory_type_ = other.memory_type_;
            allocate();
            copy_from(other);
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     * @param other Other tensor
     * @return Reference to this tensor
     */
    Tensor<T>& operator=(Tensor<T>&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            layout_ = std::move(other.layout_);
            memory_type_ = other.memory_type_;
            other.data_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get data pointer
     * @return Data pointer
     */
    T* data() const {
        // printf("[Tensor] data() called, data_=%p\n", (void*)data_);
        assert(data_ != nullptr && "Tensor::data() is nullptr!");
        return data_;
    }

    /**
     * @brief Get tensor layout
     * @return Tensor layout
     */
    const Layout& layout() const { return layout_; }

    /**
     * @brief Get tensor shape
     * @return Tensor shape
     */
    const Shape& shape() const { return layout_.shape(); }

    /**
     * @brief Get tensor stride
     * @return Tensor stride
     */
    const Stride& stride() const { return layout_.stride(); }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    dim_t ndims() const { return layout_.ndims(); }

    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    index_t numel() const { return layout_.numel(); }

    /**
     * @brief Get data type
     * @return Data type
     */
    DataType dtype() const { return get_dtype<T>(); }

    /**
     * @brief Get memory type
     * @return Memory type
     */
    MemoryType memory_type() const { return memory_type_; }

    /**
     * @brief Check if tensor is empty
     * @return true if empty, false otherwise
     */
    bool empty() const { return data_ == nullptr || layout_.numel() == 0; }

    /**
     * @brief Check if tensor is contiguous
     * @return true if contiguous, false otherwise
     */
    bool is_contiguous() const { return layout_.is_contiguous(); }

    /**
     * @brief Check if tensor is coalesced for GPU access
     * @return true if coalesced, false otherwise
     */
    bool is_coalesced() const { return layout_.is_coalesced(); }

    /**
     * @brief Access element at multi-dimensional indices
     * @param indices Multi-dimensional indices
     * @return Reference to element
     */
    template<typename... Args>
    __device__ __host__ T& operator()(Args... indices) const {
        return data_[layout_.offset(indices...)];
    }

    /**
     * @brief Get a view of this tensor
     * @return TensorView
     */
    TensorView<T> view() const {
        MemorySpace mem_space = (memory_type_ == MemoryType::DEVICE) ? 
                               MemorySpace::GLOBAL : MemorySpace::GLOBAL;
        return TensorView<T>(data_, layout_, mem_space);
    }

    /**
     * @brief Create a view of a slice
     * @param start Start dimension
     * @param end End dimension (exclusive)
     * @return Sliced view
     */
    TensorView<T> slice(dim_t start, dim_t end) const {
        return view().slice(start, end);
    }

    /**
     * @brief Create a transposed view
     * @return Transposed view
     */
    TensorView<T> transpose() const {
        return view().transpose();
    }

    /**
     * @brief Create a tiled view
     * @param tile_shape Tile shape
     * @return Tiled view
     */
    TensorView<T> tile(const Shape& tile_shape) const {
        return view().tile(tile_shape);
    }

    /**
     * @brief Create a reshaped view (requires contiguous tensor)
     * @param new_shape New shape
     * @return Reshaped view
     */
    TensorView<T> reshape(const Shape& new_shape) const {
        return view().reshape(new_shape);
    }

    /**
     * @brief Resize tensor to new shape
     * @param new_shape New shape
     */
    void resize(const Shape& new_shape) {
        if (new_shape.numel() != layout_.numel()) {
            deallocate();
            layout_ = Layout(new_shape);
            allocate();
        } else {
            layout_ = Layout(new_shape);
        }
    }

    /**
     * @brief Fill tensor with a value
     * @param value Fill value
     */
    void fill(T value) {
        // printf("[Tensor] fill: data_=%p, numel=%ld\n", (void*)data_, layout_.numel());
        assert(data_ != nullptr && "Tensor::fill: data_ is nullptr!");
        if (data_ == nullptr || layout_.numel() == 0) {
            return;
        }
        
        if (memory_type_ == MemoryType::HOST) {
            // Host memory: use std::fill
            std::fill_n(data_, layout_.numel(), value);
        } else {
            // Device memory: use cudaMemset for zero, otherwise launch a simple kernel
            if constexpr (std::is_same_v<T, __half>) {
                if (value == __float2half(0.0f)) {
                    cudaMemset(data_, 0, layout_.numel() * sizeof(T));
                } else {
                    // For non-zero values, we'd need a CUDA kernel
                    // For now, use host memory and copy
                    std::vector<T> host_data(layout_.numel(), value);
                    cudaMemcpy(data_, host_data.data(), layout_.numel() * sizeof(T), cudaMemcpyHostToDevice);
                }
            } else {
                if (value == T(0)) {
                    cudaMemset(data_, 0, layout_.numel() * sizeof(T));
                } else {
                    // For non-zero values, we'd need a CUDA kernel
                    // For now, use host memory and copy
                    std::vector<T> host_data(layout_.numel(), value);
                    cudaMemcpy(data_, host_data.data(), layout_.numel() * sizeof(T), cudaMemcpyHostToDevice);
                }
            }
        }
    }

    /**
     * @brief Copy data from host memory
     * @param host_data Host data pointer
     */
    void copy_from_host(const T* host_data) {
        if (data_ == nullptr || host_data == nullptr || layout_.numel() == 0) {
            return;
        }
        
        size_t bytes = layout_.numel() * sizeof(T);
        
        if (memory_type_ == MemoryType::HOST) {
            std::memcpy(data_, host_data, bytes);
        } else {
            cudaMemcpy(data_, host_data, bytes, cudaMemcpyHostToDevice);
        }
    }

    /**
     * @brief Copy data to host memory
     * @param host_data Host data pointer
     */
    void copy_to_host(T* host_data) const {
        if (data_ == nullptr || host_data == nullptr || layout_.numel() == 0) {
            return;
        }
        
        size_t bytes = layout_.numel() * sizeof(T);
        
        if (memory_type_ == MemoryType::HOST) {
            std::memcpy(host_data, data_, bytes);
        } else {
            cudaMemcpy(host_data, data_, bytes, cudaMemcpyDeviceToHost);
        }
    }

    /**
     * @brief Create a host tensor
     * @param shape Tensor shape
     * @return Host tensor
     */
    static Tensor<T> host(const Shape& shape) {
        return Tensor<T>(shape, MemoryType::HOST);
    }

    /**
     * @brief Create a device tensor
     * @param shape Tensor shape
     * @return Device tensor
     */
    static Tensor<T> device(const Shape& shape) {
        return Tensor<T>(shape, MemoryType::DEVICE);
    }

    /**
     * @brief Create a zero-initialized tensor
     * @param shape Tensor shape
     * @param memory_type Memory type (default: device)
     * @return Zero tensor
     */
    static Tensor<T> zeros(const Shape& shape, MemoryType memory_type = MemoryType::DEVICE) {
        Tensor<T> tensor(shape, memory_type);
        if constexpr (std::is_same_v<T, __half>) {
            tensor.fill(__float2half(0.0f));
        } else {
            tensor.fill(T(0));
        }
        return tensor;
    }

    /**
     * @brief Create a one-initialized tensor
     * @param shape Tensor shape
     * @param memory_type Memory type (default: device)
     * @return One tensor
     */
    static Tensor<T> ones(const Shape& shape, MemoryType memory_type = MemoryType::DEVICE) {
        Tensor<T> tensor(shape, memory_type);
        if constexpr (std::is_same_v<T, __half>) {
            tensor.fill(__float2half(1.0f));
        } else {
            tensor.fill(T(1));
        }
        return tensor;
    }

private:
    T* data_;
    Layout layout_;
    MemoryType memory_type_;

    /**
     * @brief Allocate memory for tensor
     */
    void allocate() {
        if (layout_.numel() == 0) {
            data_ = nullptr;
            // printf("[Tensor] allocate: numel==0, data_=nullptr\n");
            return;
        }
        
        size_t bytes = layout_.numel() * sizeof(T);
        
        if (memory_type_ == MemoryType::HOST) {
            data_ = new T[layout_.numel()];
            // printf("[Tensor] allocate: HOST, data_=%p\n", (void*)data_);
        } else if (memory_type_ == MemoryType::DEVICE) {
            cudaError_t err = cudaMalloc(&data_, bytes);
            if (err != cudaSuccess) {
                data_ = nullptr;
                fprintf(stderr, "[Tensor] allocate: cudaMalloc failed! shape=[");
                for (int i = 0; i < layout_.shape().ndims(); ++i) fprintf(stderr, "%ld,", (long)layout_.shape()[i]);
                fprintf(stderr, "] bytes=%zu, error=%s\n", bytes, cudaGetErrorString(err));
                throw std::runtime_error("Tensor::allocate: cudaMalloc failed");
            }
            // printf("[Tensor] allocate: DEVICE, data_=%p\n", (void*)data_);
        } else {
            cudaError_t err = cudaMallocManaged(&data_, bytes);
            if (err != cudaSuccess) {
                data_ = nullptr;
                fprintf(stderr, "[Tensor] allocate: cudaMallocManaged failed! shape=[");
                for (int i = 0; i < layout_.shape().ndims(); ++i) fprintf(stderr, "%ld,", (long)layout_.shape()[i]);
                fprintf(stderr, "] bytes=%zu, error=%s\n", bytes, cudaGetErrorString(err));
                throw std::runtime_error("Tensor::allocate: cudaMallocManaged failed");
            }
            // printf("[Tensor] allocate: UNIFIED, data_=%p\n", (void*)data_);
        }
    }

    /**
     * @brief Deallocate tensor memory
     */
    void deallocate() {
        if (data_ != nullptr) {
            if (memory_type_ == MemoryType::HOST) {
                delete[] data_;
            } else {
                // Both DEVICE and UNIFIED use cudaFree
                cudaFree(data_);
            }
            data_ = nullptr;
        }
    }

    /**
     * @brief Copy data from another tensor
     * @param other Source tensor
     */
    void copy_from(const Tensor<T>& other) {
        if (data_ == nullptr || other.data_ == nullptr || 
            layout_.numel() != other.layout_.numel()) {
            return;
        }
        
        size_t bytes = layout_.numel() * sizeof(T);
        
        // Determine copy type based on memory types
        cudaMemcpyKind copy_kind;
        if (memory_type_ == MemoryType::HOST && other.memory_type_ == MemoryType::HOST) {
            // Host to Host
            std::memcpy(data_, other.data_, bytes);
            return;
        } else if (memory_type_ == MemoryType::HOST && other.memory_type_ == MemoryType::DEVICE) {
            copy_kind = cudaMemcpyDeviceToHost;
        } else if (memory_type_ == MemoryType::DEVICE && other.memory_type_ == MemoryType::HOST) {
            copy_kind = cudaMemcpyHostToDevice;
        } else {
            copy_kind = cudaMemcpyDeviceToDevice;
        }
        
        cudaMemcpy(data_, other.data_, bytes, copy_kind);
    }
};

// Type aliases for common tensor types
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

} // namespace choreo_ir

#endif // CHOREO_IR_TENSOR_TENSOR_HPP 