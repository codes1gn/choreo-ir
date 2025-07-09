/**
 * @file stride.hpp
 * @brief Tensor stride representation for memory layout
 */

#ifndef CHOREO_IR_TENSOR_STRIDE_HPP
#define CHOREO_IR_TENSOR_STRIDE_HPP

#include <array>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include "shape.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {

/**
 * @class Stride
 * @brief Represents the stride (memory layout) of a tensor
 */
class Stride {
public:
    /**
     * @brief Default constructor - creates empty stride
     */
    Stride() : ndims_(0) {}

    /**
     * @brief Constructor from initializer list
     * @param strides Stride values
     */
    Stride(std::initializer_list<index_t> strides) : ndims_(static_cast<dim_t>(strides.size())) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(strides.begin(), strides.end(), strides_.begin());
    }

    /**
     * @brief Constructor from vector
     * @param strides Stride values
     */
    explicit Stride(const std::vector<index_t>& strides) : ndims_(static_cast<dim_t>(strides.size())) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(strides.begin(), strides.end(), strides_.begin());
    }

    /**
     * @brief Constructor from array
     * @param strides Stride array
     * @param ndims Number of dimensions
     */
    Stride(const index_t* strides, dim_t ndims) : ndims_(ndims) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(strides, strides + ndims, strides_.begin());
    }

    /**
     * @brief Create stride from shape (row-major layout)
     * @param shape Tensor shape
     * @return Row-major stride
     */
    static Stride from_shape_row_major(const Shape& shape) {
        if (shape.ndims() == 0) {
            return Stride();
        }
        
        Stride stride;
        stride.ndims_ = shape.ndims();
        
        // Calculate row-major strides
        stride.strides_[stride.ndims_ - 1] = 1;
        for (dim_t i = stride.ndims_ - 2; i >= 0; --i) {
            stride.strides_[i] = stride.strides_[i + 1] * shape[i + 1];
        }
        
        return stride;
    }

    /**
     * @brief Create stride from shape (column-major layout)
     * @param shape Tensor shape
     * @return Column-major stride
     */
    static Stride from_shape_col_major(const Shape& shape) {
        if (shape.ndims() == 0) {
            return Stride();
        }
        
        Stride stride;
        stride.ndims_ = shape.ndims();
        
        // Calculate column-major strides
        stride.strides_[0] = 1;
        for (dim_t i = 1; i < stride.ndims_; ++i) {
            stride.strides_[i] = stride.strides_[i - 1] * shape[i - 1];
        }
        
        return stride;
    }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    dim_t ndims() const { return ndims_; }

    /**
     * @brief Get stride at index
     * @param index Dimension index
     * @return Stride value
     */
    index_t operator[](dim_t index) const {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return strides_[index];
    }

    /**
     * @brief Get stride at index (mutable)
     * @param index Dimension index
     * @return Reference to stride value
     */
    index_t& operator[](dim_t index) {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return strides_[index];
    }

    /**
     * @brief Get stride at index with bounds checking
     * @param index Dimension index
     * @return Stride value
     */
    index_t at(dim_t index) const {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return strides_[index];
    }

    /**
     * @brief Calculate linear offset from multi-dimensional indices
     * @param indices Multi-dimensional indices
     * @return Linear offset
     */
    template<typename... Args>
    index_t offset(Args... indices) const {
        std::array<index_t, sizeof...(Args)> idx_array = {static_cast<index_t>(indices)...};
        if (sizeof...(Args) != ndims_) {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }
        
        index_t offset = 0;
        for (dim_t i = 0; i < ndims_; ++i) {
            offset += idx_array[i] * strides_[i];
        }
        return offset;
    }

    /**
     * @brief Calculate linear offset from index vector
     * @param indices Index vector
     * @return Linear offset
     */
    index_t offset(const std::vector<index_t>& indices) const {
        if (indices.size() != ndims_) {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }
        
        index_t offset = 0;
        for (dim_t i = 0; i < ndims_; ++i) {
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    /**
     * @brief Check if stride represents contiguous memory layout
     * @param shape Tensor shape
     * @return true if contiguous, false otherwise
     */
    bool is_contiguous(const Shape& shape) const {
        if (ndims_ != shape.ndims()) {
            return false;
        }
        
        if (ndims_ == 0) {
            return true;
        }
        
        // Check if it matches row-major layout
        index_t expected_stride = 1;
        for (dim_t i = ndims_ - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape[i];
        }
        
        return true;
    }

    /**
     * @brief Check if memory access is coalesced for GPU
     * @return true if coalesced, false otherwise
     */
    bool is_coalesced() const {
        // For GPU coalesced access, innermost dimension should have stride 1
        return ndims_ > 0 && strides_[ndims_ - 1] == 1;
    }

    /**
     * @brief Transpose the stride (swap last two dimensions)
     * @return Transposed stride
     */
    Stride transpose() const {
        if (ndims_ < 2) {
            return *this;
        }
        Stride result = *this;
        std::swap(result.strides_[ndims_ - 1], result.strides_[ndims_ - 2]);
        return result;
    }

    /**
     * @brief Get slice of stride
     * @param start Start dimension
     * @param end End dimension (exclusive)
     * @return Sliced stride
     */
    Stride slice(dim_t start, dim_t end) const {
        if (start >= ndims_ || end > ndims_ || start >= end) {
            throw std::out_of_range("Invalid slice range");
        }
        return Stride(strides_.data() + start, end - start);
    }

    /**
     * @brief Check if two strides are equal
     * @param other Other stride
     * @return true if equal, false otherwise
     */
    bool operator==(const Stride& other) const {
        if (ndims_ != other.ndims_) return false;
        return std::equal(strides_.begin(), strides_.begin() + ndims_, other.strides_.begin());
    }

    /**
     * @brief Check if two strides are not equal
     * @param other Other stride
     * @return true if not equal, false otherwise
     */
    bool operator!=(const Stride& other) const {
        return !(*this == other);
    }

    /**
     * @brief Get iterator to beginning
     * @return Iterator to beginning
     */
    const index_t* begin() const { return strides_.data(); }

    /**
     * @brief Get iterator to end
     * @return Iterator to end
     */
    const index_t* end() const { return strides_.data() + ndims_; }

    /**
     * @brief Convert to vector
     * @return Vector representation
     */
    std::vector<index_t> to_vector() const {
        return std::vector<index_t>(strides_.begin(), strides_.begin() + ndims_);
    }

private:
    dim_t ndims_;
    std::array<index_t, config::MAX_TENSOR_DIMS> strides_;
};

/**
 * @brief Create a stride from values
 * @param strides Stride values
 * @return Stride object
 */
template<typename... Args>
Stride make_stride(Args... strides) {
    return Stride({static_cast<index_t>(strides)...});
}

} // namespace choreo_ir

#endif // CHOREO_IR_TENSOR_STRIDE_HPP 