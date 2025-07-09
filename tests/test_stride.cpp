/**
 * @file test_stride.cpp
 * @brief Unit tests for Stride functionality
 */

#include <gtest/gtest.h>
#include "choreo-ir/tensor/stride.hpp"
#include "choreo-ir/tensor/shape.hpp"
#include <vector>
#include <stdexcept>
#include <chrono>

using namespace choreo_ir;

class StrideTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test shapes and strides
        scalar_shape = Shape();
        vector_shape = Shape({10});
        matrix_shape = Shape({5, 8});
        tensor_3d_shape = Shape({2, 3, 4});
        tensor_4d_shape = Shape({2, 3, 4, 5});

        // Row-major strides
        vector_stride_rm = Stride::from_shape_row_major(vector_shape);
        matrix_stride_rm = Stride::from_shape_row_major(matrix_shape);
        tensor_3d_stride_rm = Stride::from_shape_row_major(tensor_3d_shape);

        // Column-major strides
        matrix_stride_cm = Stride::from_shape_col_major(matrix_shape);
        tensor_3d_stride_cm = Stride::from_shape_col_major(tensor_3d_shape);
    }

    Shape scalar_shape;
    Shape vector_shape;
    Shape matrix_shape;
    Shape tensor_3d_shape;
    Shape tensor_4d_shape;

    Stride vector_stride_rm;
    Stride matrix_stride_rm;
    Stride tensor_3d_stride_rm;
    Stride matrix_stride_cm;
    Stride tensor_3d_stride_cm;
};

TEST_F(StrideTest, DefaultConstructor) {
    Stride stride;
    EXPECT_EQ(stride.ndims(), 0);
}

TEST_F(StrideTest, InitializerListConstructor) {
    Stride stride({12, 4, 1});
    EXPECT_EQ(stride.ndims(), 3);
    EXPECT_EQ(stride[0], 12);
    EXPECT_EQ(stride[1], 4);
    EXPECT_EQ(stride[2], 1);
}

TEST_F(StrideTest, VectorConstructor) {
    std::vector<index_t> strides = {20, 5, 1};
    Stride stride(strides);
    EXPECT_EQ(stride.ndims(), 3);
    EXPECT_EQ(stride[0], 20);
    EXPECT_EQ(stride[1], 5);
    EXPECT_EQ(stride[2], 1);
}

TEST_F(StrideTest, ArrayConstructor) {
    index_t strides[] = {56, 8, 1};
    Stride stride(strides, 3);
    EXPECT_EQ(stride.ndims(), 3);
    EXPECT_EQ(stride[0], 56);
    EXPECT_EQ(stride[1], 8);
    EXPECT_EQ(stride[2], 1);
}

TEST_F(StrideTest, RowMajorFromShape) {
    // Test vector
    EXPECT_EQ(vector_stride_rm.ndims(), 1);
    EXPECT_EQ(vector_stride_rm[0], 1);

    // Test matrix (5x8) - row major should be [8, 1]
    EXPECT_EQ(matrix_stride_rm.ndims(), 2);
    EXPECT_EQ(matrix_stride_rm[0], 8);
    EXPECT_EQ(matrix_stride_rm[1], 1);

    // Test 3D tensor (2x3x4) - row major should be [12, 4, 1]
    EXPECT_EQ(tensor_3d_stride_rm.ndims(), 3);
    EXPECT_EQ(tensor_3d_stride_rm[0], 12);
    EXPECT_EQ(tensor_3d_stride_rm[1], 4);
    EXPECT_EQ(tensor_3d_stride_rm[2], 1);
}

TEST_F(StrideTest, ColumnMajorFromShape) {
    // Test matrix (5x8) - column major should be [1, 5]
    EXPECT_EQ(matrix_stride_cm.ndims(), 2);
    EXPECT_EQ(matrix_stride_cm[0], 1);
    EXPECT_EQ(matrix_stride_cm[1], 5);

    // Test 3D tensor (2x3x4) - column major should be [1, 2, 6]
    EXPECT_EQ(tensor_3d_stride_cm.ndims(), 3);
    EXPECT_EQ(tensor_3d_stride_cm[0], 1);
    EXPECT_EQ(tensor_3d_stride_cm[1], 2);
    EXPECT_EQ(tensor_3d_stride_cm[2], 6);
}

TEST_F(StrideTest, ElementAccess) {
    EXPECT_EQ(matrix_stride_rm[0], 8);
    EXPECT_EQ(matrix_stride_rm[1], 1);
    EXPECT_EQ(matrix_stride_rm.at(0), 8);
    EXPECT_EQ(matrix_stride_rm.at(1), 1);

    // Test bounds checking
    EXPECT_THROW(matrix_stride_rm[2], std::out_of_range);
    EXPECT_THROW(matrix_stride_rm.at(2), std::out_of_range);
}

TEST_F(StrideTest, MutableAccess) {
    Stride stride({3, 1});
    stride[0] = 10;
    stride[1] = 2;
    EXPECT_EQ(stride[0], 10);
    EXPECT_EQ(stride[1], 2);
}

TEST_F(StrideTest, OffsetCalculation) {
    // Test matrix offset calculation (5x8 matrix with row-major stride [8, 1])
    // Element at (2, 3) should be at offset 2*8 + 3*1 = 19
    EXPECT_EQ(matrix_stride_rm.offset(2, 3), 19);
    EXPECT_EQ(matrix_stride_rm.offset(0, 0), 0);
    EXPECT_EQ(matrix_stride_rm.offset(4, 7), 39);

    // Test 3D tensor offset calculation (2x3x4 with row-major stride [12, 4, 1])
    // Element at (1, 2, 3) should be at offset 1*12 + 2*4 + 3*1 = 23
    EXPECT_EQ(tensor_3d_stride_rm.offset(1, 2, 3), 23);
    EXPECT_EQ(tensor_3d_stride_rm.offset(0, 0, 0), 0);

    // Test vector offset calculation
    std::vector<index_t> indices = {2, 3};
    EXPECT_EQ(matrix_stride_rm.offset(indices), 19);

    // Test dimension mismatch
    EXPECT_THROW(matrix_stride_rm.offset(1, 2, 3), std::invalid_argument);
    std::vector<index_t> wrong_indices = {1, 2, 3};
    EXPECT_THROW(matrix_stride_rm.offset(wrong_indices), std::invalid_argument);
}

TEST_F(StrideTest, ContiguousCheck) {
    // Row-major strides should be contiguous
    EXPECT_TRUE(matrix_stride_rm.is_contiguous(matrix_shape));
    EXPECT_TRUE(tensor_3d_stride_rm.is_contiguous(tensor_3d_shape));

    // Column-major strides should be contiguous for their respective shapes
    EXPECT_TRUE(matrix_stride_cm.is_contiguous_col_major(matrix_shape));
    EXPECT_TRUE(tensor_3d_stride_cm.is_contiguous_col_major(tensor_3d_shape));

    // Non-contiguous stride
    Stride non_contiguous({16, 2, 1});  // Wrong stride for 5x8 matrix
    EXPECT_FALSE(non_contiguous.is_contiguous(matrix_shape));

    // Test with different number of dimensions
    EXPECT_FALSE(matrix_stride_rm.is_contiguous(tensor_3d_shape));
}

TEST_F(StrideTest, CoalescedCheck) {
    // Stride with innermost dimension = 1 should be coalesced
    EXPECT_TRUE(matrix_stride_rm.is_coalesced());
    EXPECT_TRUE(tensor_3d_stride_rm.is_coalesced());

    // Stride with innermost dimension != 1 should not be coalesced
    Stride non_coalesced({8, 2});
    EXPECT_FALSE(non_coalesced.is_coalesced());

    // Empty stride
    Stride empty_stride;
    EXPECT_FALSE(empty_stride.is_coalesced());
}

TEST_F(StrideTest, Transpose) {
    // Test matrix transpose
    Stride transposed = matrix_stride_rm.transpose();
    EXPECT_EQ(transposed.ndims(), 2);
    EXPECT_EQ(transposed[0], 1);  // swapped
    EXPECT_EQ(transposed[1], 8);  // swapped

    // Test 3D tensor transpose (should swap last two dimensions)
    Stride tensor_transposed = tensor_3d_stride_rm.transpose();
    EXPECT_EQ(tensor_transposed.ndims(), 3);
    EXPECT_EQ(tensor_transposed[0], 12);
    EXPECT_EQ(tensor_transposed[1], 1);  // swapped
    EXPECT_EQ(tensor_transposed[2], 4);  // swapped

    // Test with fewer than 2 dimensions
    Stride vector_transposed = vector_stride_rm.transpose();
    EXPECT_EQ(vector_transposed.ndims(), 1);
    EXPECT_EQ(vector_transposed[0], 1);  // unchanged

    Stride scalar_stride;
    Stride scalar_transposed = scalar_stride.transpose();
    EXPECT_EQ(scalar_transposed.ndims(), 0);
}

TEST_F(StrideTest, Slice) {
    Stride tensor_4d_stride = Stride::from_shape_row_major(tensor_4d_shape);
    
    // Slice dimensions 1-3 (should get strides for dimensions 1 and 2)
    Stride sliced = tensor_4d_stride.slice(1, 3);
    EXPECT_EQ(sliced.ndims(), 2);
    EXPECT_EQ(sliced[0], tensor_4d_stride[1]);
    EXPECT_EQ(sliced[1], tensor_4d_stride[2]);

    // Test invalid slice ranges
    EXPECT_THROW(tensor_4d_stride.slice(3, 2), std::out_of_range);
    EXPECT_THROW(tensor_4d_stride.slice(0, 5), std::out_of_range);
    EXPECT_THROW(tensor_4d_stride.slice(4, 5), std::out_of_range);
}

TEST_F(StrideTest, Equality) {
    Stride same_stride = Stride::from_shape_row_major(matrix_shape);
    Stride different_stride = Stride::from_shape_col_major(matrix_shape);

    EXPECT_TRUE(matrix_stride_rm == same_stride);
    EXPECT_FALSE(matrix_stride_rm == different_stride);
    EXPECT_TRUE(matrix_stride_rm != different_stride);
    EXPECT_FALSE(matrix_stride_rm != same_stride);
}

TEST_F(StrideTest, Iterators) {
    std::vector<index_t> stride_vector;
    for (auto it = tensor_3d_stride_rm.begin(); it != tensor_3d_stride_rm.end(); ++it) {
        stride_vector.push_back(*it);
    }
    
    EXPECT_EQ(stride_vector.size(), 3);
    EXPECT_EQ(stride_vector[0], 12);
    EXPECT_EQ(stride_vector[1], 4);
    EXPECT_EQ(stride_vector[2], 1);
}

TEST_F(StrideTest, ToVector) {
    std::vector<index_t> vec = tensor_3d_stride_rm.to_vector();
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 12);
    EXPECT_EQ(vec[1], 4);
    EXPECT_EQ(vec[2], 1);
}

TEST_F(StrideTest, MakeStrideHelper) {
    Stride stride = make_stride(12, 4, 1);
    EXPECT_EQ(stride.ndims(), 3);
    EXPECT_EQ(stride[0], 12);
    EXPECT_EQ(stride[1], 4);
    EXPECT_EQ(stride[2], 1);
}

TEST_F(StrideTest, EdgeCases) {
    // Test maximum dimensions
    std::vector<index_t> max_strides(config::MAX_TENSOR_DIMS, 1);
    Stride max_stride(max_strides);
    EXPECT_EQ(max_stride.ndims(), config::MAX_TENSOR_DIMS);

    // Test too many dimensions (should throw)
    std::vector<index_t> too_many_strides(config::MAX_TENSOR_DIMS + 1, 1);
    
    // EXPECT_THROW seems to have issues, let's test manually
    bool exception_thrown = false;
    try {
        Stride test_stride(too_many_strides);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
        EXPECT_STREQ(e.what(), "Too many dimensions");
    }
    EXPECT_TRUE(exception_thrown) << "Expected std::invalid_argument to be thrown";
}

TEST_F(StrideTest, LargeStrides) {
    // Test with large stride values
    Stride large_stride({1000000, 1000, 1});
    Shape large_shape({100, 1000, 1000});
    EXPECT_TRUE(large_stride.is_contiguous(large_shape));
    EXPECT_TRUE(large_stride.is_coalesced());
}

TEST_F(StrideTest, ComplexOffsetCalculations) {
    // Test offset calculations for various tensor configurations
    
    // 4D tensor (2x3x4x5) with row-major layout
    Stride stride_4d = Stride::from_shape_row_major(tensor_4d_shape);
    // Expected strides: [60, 20, 5, 1]
    EXPECT_EQ(stride_4d[0], 60);
    EXPECT_EQ(stride_4d[1], 20);
    EXPECT_EQ(stride_4d[2], 5);
    EXPECT_EQ(stride_4d[3], 1);
    
    // Element at (1, 2, 3, 4) should be at offset 1*60 + 2*20 + 3*5 + 4*1 = 119
    EXPECT_EQ(stride_4d.offset(1, 2, 3, 4), 119);
    
    // Column-major 4D tensor
    Stride stride_4d_cm = Stride::from_shape_col_major(tensor_4d_shape);
    // Expected strides: [1, 2, 6, 24]
    EXPECT_EQ(stride_4d_cm[0], 1);
    EXPECT_EQ(stride_4d_cm[1], 2);
    EXPECT_EQ(stride_4d_cm[2], 6);
    EXPECT_EQ(stride_4d_cm[3], 24);
    
    // Element at (1, 2, 3, 4) should be at offset 1*1 + 2*2 + 3*6 + 4*24 = 119
    EXPECT_EQ(stride_4d_cm.offset(1, 2, 3, 4), 119);
}

TEST_F(StrideTest, Performance) {
    // Test performance with many offset calculations
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100000; ++i) {
        index_t offset = tensor_3d_stride_rm.offset(1, 2, 3);
        bool is_contiguous = tensor_3d_stride_rm.is_contiguous(tensor_3d_shape);
        bool is_coalesced = tensor_3d_stride_rm.is_coalesced();
        (void)offset; (void)is_contiguous; (void)is_coalesced; // Suppress unused variable warnings
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete within reasonable time (< 10ms)
    EXPECT_LT(duration.count(), 10000);
} 