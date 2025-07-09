/**
 * @file layout.hpp
 * @brief Tensor layout combining shape and stride information
 */

#ifndef CHOREO_IR_TENSOR_LAYOUT_HPP
#define CHOREO_IR_TENSOR_LAYOUT_HPP

#include "shape.hpp"
#include "stride.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"
#include <cuda_runtime.h>

namespace choreo_ir {

/**
 * @enum LayoutType
 * @brief Types of tensor memory layouts
 */
enum class LayoutType {
    ROW_MAJOR,      // C-style row-major layout
    COLUMN_MAJOR,   // Fortran-style column-major layout
    STRIDED,        // Custom strided layout
    BLOCKED         // Blocked layout for cache optimization
};

/**
 * @class Layout
 * @brief Represents the complete memory layout of a tensor
 */
class Layout {
public:
    /**
     * @brief Default constructor - creates empty layout
     */
    Layout() : shape_(), stride_(), layout_type_(LayoutType::ROW_MAJOR) {}

    /**
     * @brief Constructor from shape (row-major layout)
     * @param shape Tensor shape
     */
    explicit Layout(const Shape& shape) 
        : shape_(shape), stride_(Stride::from_shape_row_major(shape)), layout_type_(LayoutType::ROW_MAJOR) {}

    /**
     * @brief Constructor from shape and layout type
     * @param shape Tensor shape
     * @param layout_type Layout type
     */
    Layout(const Shape& shape, LayoutType layout_type) 
        : shape_(shape), layout_type_(layout_type) {
        switch (layout_type) {
            case LayoutType::ROW_MAJOR:
                stride_ = Stride::from_shape_row_major(shape);
                break;
            case LayoutType::COLUMN_MAJOR:
                stride_ = Stride::from_shape_col_major(shape);
                break;
            default:
                stride_ = Stride::from_shape_row_major(shape);
                break;
        }
    }

    /**
     * @brief Constructor from shape and stride
     * @param shape Tensor shape
     * @param stride Tensor stride
     */
    Layout(const Shape& shape, const Stride& stride) 
        : shape_(shape), stride_(stride), layout_type_(LayoutType::STRIDED) {
        if (shape.ndims() != stride.ndims()) {
            throw std::invalid_argument("Shape and stride dimensions must match");
        }
    }

    /**
     * @brief Get tensor shape
     * @return Tensor shape
     */
    const Shape& shape() const { return shape_; }

    /**
     * @brief Get tensor stride
     * @return Tensor stride
     */
    const Stride& stride() const { return stride_; }

    /**
     * @brief Get layout type
     * @return Layout type
     */
    LayoutType layout_type() const { return layout_type_; }

    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    dim_t ndims() const { return shape_.ndims(); }

    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    index_t numel() const { return shape_.numel(); }

    /**
     * @brief Get memory size in bytes
     * @param dtype Data type
     * @return Memory size in bytes
     */
    size_t memory_size(DataType dtype) const {
        return numel() * sizeof_dtype(dtype);
    }

    /**
     * @brief Check if layout is contiguous
     * @return true if contiguous, false otherwise
     *
     * For ROW_MAJOR, checks row-major contiguity;
     * For COLUMN_MAJOR, checks column-major contiguity;
     * For others, defaults to row-major.
     */
    bool is_contiguous() const {
        switch (layout_type_) {
            case LayoutType::ROW_MAJOR:
                return stride_.is_contiguous(shape_);
            case LayoutType::COLUMN_MAJOR:
                return stride_.is_contiguous_col_major(shape_);
            default:
                return stride_.is_contiguous(shape_);
        }
    }

    /**
     * @brief Check if layout is coalesced for GPU access
     * @return true if coalesced, false otherwise
     */
    bool is_coalesced() const {
        return stride_.is_coalesced();
    }

    /**
     * @brief Calculate linear offset from multi-dimensional indices
     * @param indices Multi-dimensional indices
     * @return Linear offset
     */
    template<typename... Args>
    index_t offset(Args... indices) const {
        return stride_.offset(indices...);
    }

    /**
     * @brief Calculate linear offset from index vector
     * @param indices Index vector
     * @return Linear offset
     */
    index_t offset(const std::vector<index_t>& indices) const {
        return stride_.offset(indices);
    }

    /**
     * @brief Transpose the layout (swap last two dimensions)
     * @return Transposed layout
     */
    Layout transpose() const {
        return Layout(shape_.transpose(), stride_.transpose());
    }

    /**
     * @brief Reshape the layout
     * @param new_shape New shape
     * @return Reshaped layout
     */
    Layout reshape(const Shape& new_shape) const {
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot reshape non-contiguous tensor");
        }
        if (numel() != new_shape.numel()) {
            throw std::invalid_argument("Cannot reshape: element count mismatch");
        }
        return Layout(new_shape, layout_type_);
    }

    /**
     * @brief Get slice of layout
     * @param start Start dimension
     * @param end End dimension (exclusive)
     * @return Sliced layout
     */
    Layout slice(dim_t start, dim_t end) const {
        return Layout(shape_.slice(start, end), stride_.slice(start, end));
    }

    /**
     * @brief Create a tiled layout for blocked operations
     * @param tile_shape Tile shape
     * @return Tiled layout
     */
    Layout tile(const Shape& tile_shape) const {
        if (tile_shape.ndims() != shape_.ndims()) {
            throw std::invalid_argument("Tile shape must have same number of dimensions");
        }
        
        // For now, return the original layout
        // TODO: Implement proper tiling logic
        return *this;
    }

    /**
     * @brief Check if layout is compatible with tensor core operations
     * @return true if compatible, false otherwise
     */
    bool is_tensor_core_compatible() const {
        // Check if the layout is suitable for tensor core operations
        // Tensor core typically requires specific alignment and dimensions
        if (shape_.ndims() < 2) {
            return false;
        }
        
        // Check if last two dimensions are compatible with WMMA
        index_t m = shape_.second_last_dim();
        index_t n = shape_.last_dim();
        
        return (m % config::WMMA_M == 0) && (n % config::WMMA_N == 0) && is_coalesced();
    }

    /**
     * @brief Get optimal block size for CUDA kernel launch
     * @return Optimal block size
     */
    dim3 get_optimal_block_size() const {
        if (shape_.ndims() == 0) {
            return dim3(1, 1, 1);
        }
        
        // For 1D tensors
        if (shape_.ndims() == 1) {
            return dim3(std::min(static_cast<index_t>(config::DEFAULT_BLOCK_SIZE), shape_[0]), 1, 1);
        }
        
        // For 2D tensors (matrices)
        if (shape_.ndims() == 2) {
            index_t total_threads = config::DEFAULT_BLOCK_SIZE;
            index_t block_x = std::min(static_cast<index_t>(32), shape_[1]);  // Coalesced access
            index_t block_y = std::min(total_threads / block_x, shape_[0]);
            return dim3(block_x, block_y, 1);
        }
        
        // For higher dimensional tensors, use 1D block
        return dim3(config::DEFAULT_BLOCK_SIZE, 1, 1);
    }

    /**
     * @brief Get grid size for CUDA kernel launch
     * @return Grid size
     */
    dim3 get_grid_size() const {
        dim3 block_size = get_optimal_block_size();
        
        if (shape_.ndims() == 0) {
            return dim3(1, 1, 1);
        }
        
        // For 1D tensors
        if (shape_.ndims() == 1) {
            index_t grid_x = (shape_[0] + block_size.x - 1) / block_size.x;
            return dim3(grid_x, 1, 1);
        }
        
        // For 2D tensors
        if (shape_.ndims() == 2) {
            index_t grid_x = (shape_[1] + block_size.x - 1) / block_size.x;
            index_t grid_y = (shape_[0] + block_size.y - 1) / block_size.y;
            return dim3(grid_x, grid_y, 1);
        }
        
        // For higher dimensional tensors
        index_t total_elements = numel();
        index_t total_threads_per_block = block_size.x * block_size.y * block_size.z;
        index_t grid_x = (total_elements + total_threads_per_block - 1) / total_threads_per_block;
        return dim3(grid_x, 1, 1);
    }

    /**
     * @brief Check if two layouts are equal
     * @param other Other layout
     * @return true if equal, false otherwise
     */
    bool operator==(const Layout& other) const {
        return shape_ == other.shape_ && stride_ == other.stride_;
    }

    /**
     * @brief Check if two layouts are not equal
     * @param other Other layout
     * @return true if not equal, false otherwise
     */
    bool operator!=(const Layout& other) const {
        return !(*this == other);
    }

private:
    Shape shape_;
    Stride stride_;
    LayoutType layout_type_;
};

/**
 * @brief Create a row-major layout from shape
 * @param shape Tensor shape
 * @return Row-major layout
 */
inline Layout make_row_major_layout(const Shape& shape) {
    return Layout(shape, LayoutType::ROW_MAJOR);
}

/**
 * @brief Create a column-major layout from shape
 * @param shape Tensor shape
 * @return Column-major layout
 */
inline Layout make_col_major_layout(const Shape& shape) {
    return Layout(shape, LayoutType::COLUMN_MAJOR);
}

/**
 * @brief Create a strided layout from shape and stride
 * @param shape Tensor shape
 * @param stride Tensor stride
 * @return Strided layout
 */
inline Layout make_strided_layout(const Shape& shape, const Stride& stride) {
    return Layout(shape, stride);
}

} // namespace choreo_ir

#endif // CHOREO_IR_TENSOR_LAYOUT_HPP 