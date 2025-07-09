/**
 * @file tensor.hpp
 * @brief Core tensor class with zero-cost abstractions and memory hierarchy support
 */

#ifndef CHOREO_IR_TENSOR_TENSOR_HPP
#define CHOREO_IR_TENSOR_TENSOR_HPP

#include <memory>
#include <cuda_runtime.h>
#include "layout.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"
#include "../core/device.hpp"

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
     * @brief Copy data to shared memory (device function)
     * @param shared_ptr Shared memory pointer
     * @param tile_shape Shape of the tile to copy
     * @return TensorView pointing to shared memory
     */
    template<dim_t MAX_SHARED_SIZE = 49152> // 48KB default shared memory
    __device__ TensorView<T> copy_to_shared(T* shared_ptr, const Shape& tile_shape) const {
        static_assert(MAX_SHARED_SIZE > 0, "Shared memory size must be positive");
        
        // Calculate thread mapping for coalesced access
        index_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        index_t total_threads = blockDim.x * blockDim.y * blockDim.z;
        index_t total_elements = tile_shape.numel();
        
        // Each thread copies multiple elements if needed
        for (index_t i = tid; i < total_elements; i += total_threads) {
            shared_ptr[i] = data_[i];
        }
        
        __syncthreads(); // Ensure all threads complete the copy
        
        Layout shared_layout(tile_shape);
        return TensorView<T>(shared_ptr, shared_layout, MemorySpace::SHARED);
    }

    /**
     * @brief Copy data to local memory (registers)
     * @param local_array Local array to copy to
     * @param num_elements Number of elements to copy
     * @return TensorView pointing to local memory
     */
    template<dim_t LOCAL_SIZE>
    __device__ TensorView<T> copy_to_local(T (&local_array)[LOCAL_SIZE]) const {
        static_assert(LOCAL_SIZE > 0, "Local array size must be positive");
        
        index_t copy_size = min(LOCAL_SIZE, static_cast<dim_t>(numel()));
        
        #pragma unroll
        for (dim_t i = 0; i < copy_size; ++i) {
            local_array[i] = data_[i];
        }
        
        Shape local_shape({copy_size});
        Layout local_layout(local_shape);
        return TensorView<T>(local_array, local_layout, MemorySpace::LOCAL);
    }

    /**
     * @brief Asynchronous copy from global to shared memory
     * @param shared_ptr Shared memory destination
     * @param tile_shape Shape of tile to copy
     * @param stream CUDA stream for async copy
     * @return TensorView pointing to shared memory
     */
    __device__ TensorView<T> async_copy_to_shared(T* shared_ptr, const Shape& tile_shape, 
                                                 cudaStream_t stream = 0) const {
        // Use cp.async instructions if available (Ampere+)
        #if __CUDA_ARCH__ >= 800
            // Implementation would use cp.async for memory-level parallelism
            return copy_to_shared(shared_ptr, tile_shape);
        #else
            return copy_to_shared(shared_ptr, tile_shape);
        #endif
    }

    /**
     * @brief Assignment operator for data transfer: dst = src.tile(shape)
     * This is the core abstraction for expressing data movement
     */
    template<typename SrcT>
    __device__ TensorView<T>& operator=(const TensorView<SrcT>& src) {
        // Type checking
        static_assert(std::is_convertible_v<SrcT, T>, "Source type must be convertible to destination type");
        
        // Memory space transfer logic
        if (memory_space_ == MemorySpace::SHARED && src.memory_space() == MemorySpace::GLOBAL) {
            // Global -> Shared transfer
            return *this = src.copy_to_shared(data_, layout_.shape());
        } else if (memory_space_ == MemorySpace::LOCAL && src.memory_space() == MemorySpace::SHARED) {
            // Shared -> Local transfer (would need proper local array handling)
            // This is conceptual - actual implementation would be more complex
        } else {
            // Direct copy for same memory space
            index_t min_elements = min(numel(), src.numel());
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
 * @brief Thread-local tensor stored in registers
 */
template<typename T, dim_t SIZE>
class LocalTensor {
public:
    __device__ LocalTensor() = default;
    
    __device__ LocalTensor(const Shape& shape) : layout_(shape) {
        static_assert(SIZE >= 1, "Local tensor size must be at least 1");
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
 * @class SharedTensor
 * @brief Block-level shared memory tensor
 */
template<typename T, dim_t SIZE>
class SharedTensor {
public:
    __device__ SharedTensor() = default;
    
    __device__ SharedTensor(const Shape& shape) : layout_(shape) {
        static_assert(SIZE >= 1, "Shared tensor size must be at least 1");
    }
    
    __device__ TensorView<T> view() {
        return TensorView<T>(data_, layout_, MemorySpace::SHARED);
    }
    
    __device__ const TensorView<T> view() const {
        return TensorView<T>(const_cast<T*>(data_), layout_, MemorySpace::SHARED);
    }
    
    __device__ T& operator[](index_t i) { return data_[i]; }
    __device__ const T& operator[](index_t i) const { return data_[i]; }
    
private:
    __shared__ static T data_[SIZE];
    Layout layout_;
};

// Static member definition
template<typename T, dim_t SIZE>
__shared__ T SharedTensor<T, SIZE>::data_[SIZE];

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
    Tensor() : data_(nullptr), layout_(), memory_type_(MemoryType::DEVICE) {}

    /**
     * @brief Constructor from shape
     * @param shape Tensor shape
     * @param memory_type Memory type (default: device)
     */
    explicit Tensor(const Shape& shape, MemoryType memory_type = MemoryType::DEVICE)
        : layout_(shape), memory_type_(memory_type) {
        allocate();
    }

    /**
     * @brief Constructor from layout
     * @param layout Tensor layout
     * @param memory_type Memory type (default: device)
     */
    explicit Tensor(const Layout& layout, MemoryType memory_type = MemoryType::DEVICE)
        : layout_(layout), memory_type_(memory_type) {
        allocate();
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
    T* data() const { return data_; }

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
        // Implementation would use CUDA kernels for device memory
        // or std::fill for host memory
    }

    /**
     * @brief Copy data from host memory
     * @param host_data Host data pointer
     */
    void copy_from_host(const T* host_data) {
        // Implementation would use cudaMemcpy
    }

    /**
     * @brief Copy data to host memory
     * @param host_data Host data pointer
     */
    void copy_to_host(T* host_data) const {
        // Implementation would use cudaMemcpy
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
        tensor.fill(T(0));
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
        tensor.fill(T(1));
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
        // Implementation would handle different memory types
    }

    /**
     * @brief Deallocate tensor memory
     */
    void deallocate() {
        // Implementation would handle different memory types
    }

    /**
     * @brief Copy data from another tensor
     * @param other Source tensor
     */
    void copy_from(const Tensor<T>& other) {
        // Implementation would use appropriate copy method
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