/**
 * @file test_layout.cpp
 * @brief Comprehensive unit tests for Layout class
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include <chrono>
#include "choreo-ir/tensor/layout.hpp"

using namespace choreo_ir;

class LayoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test data
        shape_2d_ = Shape({4, 6});
        shape_3d_ = Shape({2, 3, 4});
        shape_1d_ = Shape({24});
    }

    Shape shape_2d_;
    Shape shape_3d_;
    Shape shape_1d_;
};

// Test constructors
TEST_F(LayoutTest, DefaultConstructor) {
    Layout layout;
    
    EXPECT_EQ(layout.ndims(), 0);
    EXPECT_EQ(layout.numel(), 1);  // Default shape has numel = 1
    EXPECT_EQ(layout.layout_type(), LayoutType::ROW_MAJOR);
}

TEST_F(LayoutTest, ConstructorFromShape) {
    Layout layout(shape_2d_);
    
    EXPECT_EQ(layout.shape(), shape_2d_);
    EXPECT_EQ(layout.ndims(), 2);
    EXPECT_EQ(layout.numel(), 24);
    EXPECT_EQ(layout.layout_type(), LayoutType::ROW_MAJOR);
    EXPECT_TRUE(layout.is_contiguous());
}

TEST_F(LayoutTest, ConstructorFromShapeAndLayoutType) {
    Layout row_major_layout(shape_2d_, LayoutType::ROW_MAJOR);
    Layout col_major_layout(shape_2d_, LayoutType::COLUMN_MAJOR);
    
    EXPECT_EQ(row_major_layout.layout_type(), LayoutType::ROW_MAJOR);
    EXPECT_EQ(col_major_layout.layout_type(), LayoutType::COLUMN_MAJOR);
    
    // Both should be contiguous but have different strides
    EXPECT_TRUE(row_major_layout.is_contiguous());
    EXPECT_TRUE(col_major_layout.is_contiguous());
    EXPECT_NE(row_major_layout.stride(), col_major_layout.stride());
}

TEST_F(LayoutTest, ConstructorFromShapeAndStride) {
    Stride custom_stride({12, 2});  // Custom stride pattern
    Layout layout(shape_2d_, custom_stride);
    
    EXPECT_EQ(layout.shape(), shape_2d_);
    EXPECT_EQ(layout.stride(), custom_stride);
    EXPECT_EQ(layout.layout_type(), LayoutType::STRIDED);
}

TEST_F(LayoutTest, ConstructorDimensionMismatch) {
    Shape shape({2, 3});
    Stride stride({4, 2, 1});  // Different number of dimensions
    
    bool exception_thrown = false;
    try {
        Layout layout(shape, stride);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    EXPECT_TRUE(exception_thrown);
}

// Test basic properties
TEST_F(LayoutTest, BasicProperties) {
    Layout layout(shape_3d_);
    
    EXPECT_EQ(layout.ndims(), 3);
    EXPECT_EQ(layout.numel(), 24);
    EXPECT_EQ(layout.memory_size(DataType::FLOAT32), 24 * 4);  // 24 elements * 4 bytes
    EXPECT_EQ(layout.memory_size(DataType::FLOAT16), 24 * 2);  // 24 elements * 2 bytes
}

// Test offset calculation
TEST_F(LayoutTest, OffsetCalculation) {
    Layout row_major_layout(shape_2d_, LayoutType::ROW_MAJOR);
    Layout col_major_layout(shape_2d_, LayoutType::COLUMN_MAJOR);
    
    // Test multi-argument offset
    index_t row_major_offset = row_major_layout.offset(1, 2);
    index_t col_major_offset = col_major_layout.offset(1, 2);
    
    EXPECT_EQ(row_major_offset, 1 * 6 + 2);  // Row-major: row * cols + col
    EXPECT_EQ(col_major_offset, 2 * 4 + 1);  // Column-major: col * rows + row
    
    // Test vector offset
    std::vector<index_t> indices = {1, 2};
    EXPECT_EQ(row_major_layout.offset(indices), row_major_offset);
    EXPECT_EQ(col_major_layout.offset(indices), col_major_offset);
}

// Test layout operations
TEST_F(LayoutTest, TransposeOperation) {
    Layout original(shape_2d_);
    Layout transposed = original.transpose();
    
    // Shape should be transposed
    EXPECT_EQ(transposed.shape()[0], shape_2d_[1]);
    EXPECT_EQ(transposed.shape()[1], shape_2d_[0]);
    EXPECT_EQ(transposed.numel(), original.numel());
}

TEST_F(LayoutTest, ReshapeOperation) {
    Layout original(shape_1d_);
    Shape new_shape({4, 6});
    
    Layout reshaped = original.reshape(new_shape);
    
    EXPECT_EQ(reshaped.shape(), new_shape);
    EXPECT_EQ(reshaped.numel(), original.numel());
}

TEST_F(LayoutTest, ReshapeNonContiguous) {
    // Create non-contiguous layout
    Stride non_contiguous_stride({12, 1});  // Skip elements
    Layout non_contiguous(shape_2d_, non_contiguous_stride);
    
    bool exception_thrown = false;
    try {
        non_contiguous.reshape(Shape({24}));
    } catch (const std::runtime_error& e) {
        exception_thrown = true;
    }
    EXPECT_TRUE(exception_thrown);
}

TEST_F(LayoutTest, ReshapeElementCountMismatch) {
    Layout original(shape_2d_);
    Shape wrong_shape({3, 7});  // 21 elements vs 24
    
    bool exception_thrown = false;
    try {
        original.reshape(wrong_shape);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    EXPECT_TRUE(exception_thrown);
}

TEST_F(LayoutTest, SliceOperation) {
    Layout original(shape_3d_);
    Layout sliced = original.slice(1, 3);  // Take dimensions 1 and 2
    
    EXPECT_EQ(sliced.ndims(), 2);
    EXPECT_EQ(sliced.shape()[0], shape_3d_[1]);
    EXPECT_EQ(sliced.shape()[1], shape_3d_[2]);
}

// Test tiling operation
TEST_F(LayoutTest, TileOperation) {
    Layout original(shape_2d_);
    Shape tile_shape({2, 3});
    
    Layout tiled = original.tile(tile_shape);
    
    // Current implementation returns original layout
    // TODO: Implement proper tiling logic
    EXPECT_EQ(tiled.shape(), original.shape());
}

TEST_F(LayoutTest, TileDimensionMismatch) {
    Layout original(shape_2d_);
    Shape wrong_tile_shape({2, 3, 4});  // Different number of dimensions
    
    bool exception_thrown = false;
    try {
        original.tile(wrong_tile_shape);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    EXPECT_TRUE(exception_thrown);
}

// Test tensor core compatibility
TEST_F(LayoutTest, TensorCoreCompatibility) {
    // Compatible shapes (multiple of 16)
    Shape compatible_shape({32, 48});
    Layout compatible_layout(compatible_shape);
    EXPECT_TRUE(compatible_layout.is_tensor_core_compatible());
    
    // Incompatible shapes
    Shape incompatible_shape({30, 45});
    Layout incompatible_layout(incompatible_shape);
    EXPECT_FALSE(incompatible_layout.is_tensor_core_compatible());
    
    // 1D tensor should not be compatible
    Shape shape_1d({256});
    Layout layout_1d(shape_1d);
    EXPECT_FALSE(layout_1d.is_tensor_core_compatible());
}

// Test CUDA kernel configuration
TEST_F(LayoutTest, OptimalBlockSize) {
    // Test 1D tensor
    Shape shape_1d({1024});
    Layout layout_1d(shape_1d);
    dim3 block_1d = layout_1d.get_optimal_block_size();
    EXPECT_LE(block_1d.x, 1024u);
    EXPECT_EQ(block_1d.y, 1u);
    EXPECT_EQ(block_1d.z, 1u);
    
    // Test 2D tensor
    Layout layout_2d(shape_2d_);
    dim3 block_2d = layout_2d.get_optimal_block_size();
    EXPECT_LE(block_2d.x, 32u);  // Coalesced access
    EXPECT_GE(block_2d.y, 1u);
    EXPECT_EQ(block_2d.z, 1u);
    
    // Test empty tensor
    Layout empty_layout;
    dim3 block_empty = empty_layout.get_optimal_block_size();
    EXPECT_EQ(block_empty.x, 1u);
    EXPECT_EQ(block_empty.y, 1u);
    EXPECT_EQ(block_empty.z, 1u);
}

TEST_F(LayoutTest, GridSize) {
    Layout layout(shape_2d_);
    dim3 block_size = layout.get_optimal_block_size();
    dim3 grid_size = layout.get_grid_size();
    
    // Grid should be large enough to cover all elements
    EXPECT_GE(grid_size.x * block_size.x, static_cast<unsigned int>(shape_2d_[1]));
    EXPECT_GE(grid_size.y * block_size.y, static_cast<unsigned int>(shape_2d_[0]));
}

// Test equality operators
TEST_F(LayoutTest, EqualityOperators) {
    Layout layout1(shape_2d_);
    Layout layout2(shape_2d_);
    Layout layout3(shape_3d_);
    
    EXPECT_TRUE(layout1 == layout2);
    EXPECT_FALSE(layout1 != layout2);
    EXPECT_FALSE(layout1 == layout3);
    EXPECT_TRUE(layout1 != layout3);
}

// Test factory functions
TEST_F(LayoutTest, FactoryFunctions) {
    Layout row_major = make_row_major_layout(shape_2d_);
    Layout col_major = make_col_major_layout(shape_2d_);
    
    EXPECT_EQ(row_major.layout_type(), LayoutType::ROW_MAJOR);
    EXPECT_EQ(col_major.layout_type(), LayoutType::COLUMN_MAJOR);
    EXPECT_TRUE(row_major.is_contiguous());
    EXPECT_TRUE(col_major.is_contiguous());
}

// Test contiguity checking
TEST_F(LayoutTest, ContiguityChecking) {
    // Contiguous layouts
    Layout row_major(shape_2d_, LayoutType::ROW_MAJOR);
    Layout col_major(shape_2d_, LayoutType::COLUMN_MAJOR);
    EXPECT_TRUE(row_major.is_contiguous());
    EXPECT_TRUE(col_major.is_contiguous());
    
    // Non-contiguous layout
    Stride non_contiguous_stride({12, 2});  // Skip elements
    Layout non_contiguous(shape_2d_, non_contiguous_stride);
    EXPECT_FALSE(non_contiguous.is_contiguous());
}

// Test coalescing for GPU access
TEST_F(LayoutTest, CoalescingCheck) {
    Layout row_major(shape_2d_, LayoutType::ROW_MAJOR);
    
    // Row-major should be coalesced (stride[last_dim] == 1)
    EXPECT_TRUE(row_major.is_coalesced());
    
    // Create non-coalesced layout
    Stride non_coalesced_stride({1, 4});  // Column-major with gaps
    Layout non_coalesced(shape_2d_, non_coalesced_stride);
    EXPECT_FALSE(non_coalesced.is_coalesced());
}

// Test complex operations
TEST_F(LayoutTest, ComplexOperations) {
    // Create a complex layout and perform multiple operations
    Layout original(shape_3d_);
    
    // Chain operations
    Layout processed = original
        .slice(0, 2)      // Take first 2 dimensions
        .transpose();     // Transpose
    
    EXPECT_EQ(processed.ndims(), 2);
    EXPECT_EQ(processed.shape()[0], shape_3d_[1]);
    EXPECT_EQ(processed.shape()[1], shape_3d_[0]);
}

// Performance test for common operations
TEST_F(LayoutTest, PerformanceTest) {
    Shape large_shape({1024, 1024});
    Layout large_layout(large_shape);
    
    // Time-critical operations should be fast
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10000; ++i) {
        volatile index_t offset = large_layout.offset(512, 512);
        (void)offset;  // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete in reasonable time (less than 10ms for 10k operations)
    EXPECT_LT(duration.count(), 10000);
}

// Edge cases
TEST_F(LayoutTest, EdgeCases) {
    // Very large dimensions (but within limits)
    Shape large_shape({1000000});
    Layout large_layout(large_shape);
    EXPECT_EQ(large_layout.numel(), 1000000);
    
    // Single element tensor
    Shape single_element({1});
    Layout single_layout(single_element);
    EXPECT_EQ(single_layout.numel(), 1);
    EXPECT_TRUE(single_layout.is_contiguous());
    
    // Maximum dimensions
    std::vector<index_t> max_dims(config::MAX_TENSOR_DIMS, 2);
    Shape max_shape(max_dims);
    Layout max_layout(max_shape);
    EXPECT_EQ(max_layout.ndims(), config::MAX_TENSOR_DIMS);
} 