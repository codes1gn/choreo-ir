/**
 * @file shape.hpp
 * @brief Tensor shape representation and operations
 */

#ifndef CHOREO_IR_TENSOR_SHAPE_HPP
#define CHOREO_IR_TENSOR_SHAPE_HPP

#include <array>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {

/**
 * @class Shape
 * @brief Represents the shape (dimensions) of a tensor
 */
class Shape {
public:
    /**
     * @brief Default constructor - creates a scalar shape
     */
    Shape() : ndims_(0) {}

    /**
     * @brief Constructor from initializer list
     * @param dims Dimensions
     */
    Shape(std::initializer_list<index_t> dims) : ndims_(static_cast<dim_t>(dims.size())) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(dims.begin(), dims.end(), dims_.begin());
    }

    /**
     * @brief Constructor from vector
     * @param dims Dimensions
     */
    explicit Shape(const std::vector<index_t>& dims) : ndims_(static_cast<dim_t>(dims.size())) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(dims.begin(), dims.end(), dims_.begin());
    }

    /**
     * @brief Constructor from array
     * @param dims Dimensions array
     * @param ndims Number of dimensions
     */
    Shape(const index_t* dims, dim_t ndims) : ndims_(ndims) {
        if (ndims_ > config::MAX_TENSOR_DIMS) {
            throw std::invalid_argument("Too many dimensions");
        }
        std::copy(dims, dims + ndims, dims_.begin());
    }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    dim_t ndims() const { return ndims_; }

    /**
     * @brief Get dimension at index
     * @param index Dimension index
     * @return Dimension size
     */
    index_t operator[](dim_t index) const {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return dims_[index];
    }

    /**
     * @brief Get dimension at index (mutable)
     * @param index Dimension index
     * @return Reference to dimension size
     */
    index_t& operator[](dim_t index) {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return dims_[index];
    }

    /**
     * @brief Get dimension at index with bounds checking
     * @param index Dimension index
     * @return Dimension size
     */
    index_t at(dim_t index) const {
        if (index >= ndims_) {
            throw std::out_of_range("Dimension index out of range");
        }
        return dims_[index];
    }

    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    index_t numel() const {
        return std::accumulate(dims_.begin(), dims_.begin() + ndims_, 1LL, std::multiplies<index_t>());
    }

    /**
     * @brief Check if shape is empty (any dimension is 0)
     * @return true if empty, false otherwise
     */
    bool empty() const {
        return std::any_of(dims_.begin(), dims_.begin() + ndims_, [](index_t dim) { return dim == 0; });
    }

    /**
     * @brief Check if shape is scalar (0 dimensions)
     * @return true if scalar, false otherwise
     */
    bool is_scalar() const { return ndims_ == 0; }

    /**
     * @brief Check if shape is vector (1 dimension)
     * @return true if vector, false otherwise
     */
    bool is_vector() const { return ndims_ == 1; }

    /**
     * @brief Check if shape is matrix (2 dimensions)
     * @return true if matrix, false otherwise
     */
    bool is_matrix() const { return ndims_ == 2; }

    /**
     * @brief Get the last dimension (commonly used for matrix operations)
     * @return Last dimension size
     */
    index_t last_dim() const {
        return ndims_ > 0 ? dims_[ndims_ - 1] : 1;
    }

    /**
     * @brief Get the second to last dimension
     * @return Second to last dimension size
     */
    index_t second_last_dim() const {
        return ndims_ > 1 ? dims_[ndims_ - 2] : 1;
    }

    /**
     * @brief Reshape to new dimensions
     * @param new_shape New shape
     * @return New shape object
     */
    Shape reshape(const Shape& new_shape) const {
        if (numel() != new_shape.numel()) {
            throw std::invalid_argument("Cannot reshape: element count mismatch");
        }
        return new_shape;
    }

    /**
     * @brief Transpose the shape (swap last two dimensions)
     * @return Transposed shape
     */
    Shape transpose() const {
        if (ndims_ < 2) {
            return *this;
        }
        Shape result = *this;
        std::swap(result.dims_[ndims_ - 1], result.dims_[ndims_ - 2]);
        return result;
    }

    /**
     * @brief Get slice of shape
     * @param start Start dimension
     * @param end End dimension (exclusive)
     * @return Sliced shape
     */
    Shape slice(dim_t start, dim_t end) const {
        if (start >= ndims_ || end > ndims_ || start >= end) {
            throw std::out_of_range("Invalid slice range");
        }
        return Shape(dims_.data() + start, end - start);
    }

    /**
     * @brief Check if two shapes are equal
     * @param other Other shape
     * @return true if equal, false otherwise
     */
    bool operator==(const Shape& other) const {
        if (ndims_ != other.ndims_) return false;
        return std::equal(dims_.begin(), dims_.begin() + ndims_, other.dims_.begin());
    }

    /**
     * @brief Check if two shapes are not equal
     * @param other Other shape
     * @return true if not equal, false otherwise
     */
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }

    /**
     * @brief Get iterator to beginning
     * @return Iterator to beginning
     */
    const index_t* begin() const { return dims_.data(); }

    /**
     * @brief Get iterator to end
     * @return Iterator to end
     */
    const index_t* end() const { return dims_.data() + ndims_; }

    /**
     * @brief Convert to vector
     * @return Vector representation
     */
    std::vector<index_t> to_vector() const {
        return std::vector<index_t>(dims_.begin(), dims_.begin() + ndims_);
    }

private:
    dim_t ndims_;
    std::array<index_t, config::MAX_TENSOR_DIMS> dims_;
};

/**
 * @brief Create a shape from dimensions
 * @param dims Dimensions
 * @return Shape object
 */
template<typename... Args>
Shape make_shape(Args... dims) {
    return Shape({static_cast<index_t>(dims)...});
}

} // namespace choreo_ir

#endif // CHOREO_IR_TENSOR_SHAPE_HPP 