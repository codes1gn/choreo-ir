/**
 * @file test_shape.cpp
 * @brief Unit tests for Shape functionality
 */

#include <gtest/gtest.h>
#include "choreo-ir/tensor/shape.hpp"
#include <vector>
#include <stdexcept>
#include <chrono>
#include <iostream>

using namespace choreo_ir;

class ShapeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test shapes
        scalar_shape = Shape();
        vector_shape = Shape({10});
        matrix_shape = Shape({5, 8});
        tensor_3d_shape = Shape({2, 3, 4});
        tensor_4d_shape = Shape({2, 3, 4, 5});
    }

    Shape scalar_shape;
    Shape vector_shape;
    Shape matrix_shape;
    Shape tensor_3d_shape;
    Shape tensor_4d_shape;
};

TEST_F(ShapeTest, DefaultConstructor) {
    Shape shape;
    EXPECT_EQ(shape.ndims(), 0);
    EXPECT_TRUE(shape.is_scalar());
    EXPECT_FALSE(shape.is_vector());
    EXPECT_FALSE(shape.is_matrix());
    EXPECT_EQ(shape.numel(), 1);
}

TEST_F(ShapeTest, InitializerListConstructor) {
    Shape shape({3, 4, 5});
    EXPECT_EQ(shape.ndims(), 3);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 5);
    EXPECT_EQ(shape.numel(), 60);
}

TEST_F(ShapeTest, VectorConstructor) {
    std::vector<index_t> dims = {2, 3, 4, 5};
    Shape shape(dims);
    EXPECT_EQ(shape.ndims(), 4);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape[3], 5);
    EXPECT_EQ(shape.numel(), 120);
}

TEST_F(ShapeTest, ArrayConstructor) {
    index_t dims[] = {6, 7, 8};
    Shape shape(dims, 3);
    EXPECT_EQ(shape.ndims(), 3);
    EXPECT_EQ(shape[0], 6);
    EXPECT_EQ(shape[1], 7);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape.numel(), 336);
}

TEST_F(ShapeTest, ElementAccess) {
    EXPECT_EQ(matrix_shape[0], 5);
    EXPECT_EQ(matrix_shape[1], 8);
    EXPECT_EQ(matrix_shape.at(0), 5);
    EXPECT_EQ(matrix_shape.at(1), 8);
    
    // Test bounds checking
    EXPECT_THROW(matrix_shape[2], std::out_of_range);
    EXPECT_THROW(matrix_shape.at(2), std::out_of_range);
}

TEST_F(ShapeTest, MutableAccess) {
    Shape shape({3, 4});
    shape[0] = 10;
    shape[1] = 20;
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
    EXPECT_EQ(shape.numel(), 200);
}

TEST_F(ShapeTest, ShapeProperties) {
    EXPECT_TRUE(scalar_shape.is_scalar());
    EXPECT_FALSE(scalar_shape.is_vector());
    EXPECT_FALSE(scalar_shape.is_matrix());

    EXPECT_FALSE(vector_shape.is_scalar());
    EXPECT_TRUE(vector_shape.is_vector());
    EXPECT_FALSE(vector_shape.is_matrix());

    EXPECT_FALSE(matrix_shape.is_scalar());
    EXPECT_FALSE(matrix_shape.is_vector());
    EXPECT_TRUE(matrix_shape.is_matrix());

    EXPECT_FALSE(tensor_3d_shape.is_scalar());
    EXPECT_FALSE(tensor_3d_shape.is_vector());
    EXPECT_FALSE(tensor_3d_shape.is_matrix());
}

TEST_F(ShapeTest, DimensionAccess) {
    EXPECT_EQ(scalar_shape.last_dim(), 1);
    EXPECT_EQ(scalar_shape.second_last_dim(), 1);

    EXPECT_EQ(vector_shape.last_dim(), 10);
    EXPECT_EQ(vector_shape.second_last_dim(), 1);

    EXPECT_EQ(matrix_shape.last_dim(), 8);
    EXPECT_EQ(matrix_shape.second_last_dim(), 5);

    EXPECT_EQ(tensor_3d_shape.last_dim(), 4);
    EXPECT_EQ(tensor_3d_shape.second_last_dim(), 3);
}

TEST_F(ShapeTest, NumelCalculation) {
    EXPECT_EQ(scalar_shape.numel(), 1);
    EXPECT_EQ(vector_shape.numel(), 10);
    EXPECT_EQ(matrix_shape.numel(), 40);
    EXPECT_EQ(tensor_3d_shape.numel(), 24);
    EXPECT_EQ(tensor_4d_shape.numel(), 120);
}

TEST_F(ShapeTest, EmptyCheck) {
    Shape empty_shape({0, 5});
    EXPECT_TRUE(empty_shape.empty());
    EXPECT_FALSE(matrix_shape.empty());
    
    Shape zero_dim_shape({3, 0, 4});
    EXPECT_TRUE(zero_dim_shape.empty());
}

TEST_F(ShapeTest, Reshape) {
    Shape reshaped = matrix_shape.reshape(Shape({8, 5}));
    EXPECT_EQ(reshaped.ndims(), 2);
    EXPECT_EQ(reshaped[0], 8);
    EXPECT_EQ(reshaped[1], 5);
    EXPECT_EQ(reshaped.numel(), 40);

    // Test reshape with different total elements (should throw)
    EXPECT_THROW(matrix_shape.reshape(Shape({3, 4})), std::invalid_argument);
}

TEST_F(ShapeTest, Transpose) {
    Shape transposed = matrix_shape.transpose();
    EXPECT_EQ(transposed.ndims(), 2);
    EXPECT_EQ(transposed[0], 8);
    EXPECT_EQ(transposed[1], 5);

    Shape scalar_transposed = scalar_shape.transpose();
    EXPECT_EQ(scalar_transposed.ndims(), 0);

    Shape vector_transposed = vector_shape.transpose();
    EXPECT_EQ(vector_transposed.ndims(), 1);
    EXPECT_EQ(vector_transposed[0], 10);

    Shape tensor_transposed = tensor_3d_shape.transpose();
    EXPECT_EQ(tensor_transposed.ndims(), 3);
    EXPECT_EQ(tensor_transposed[0], 2);
    EXPECT_EQ(tensor_transposed[1], 4);  // swapped
    EXPECT_EQ(tensor_transposed[2], 3);  // swapped
}

TEST_F(ShapeTest, Slice) {
    Shape sliced = tensor_4d_shape.slice(1, 3);
    EXPECT_EQ(sliced.ndims(), 2);
    EXPECT_EQ(sliced[0], 3);
    EXPECT_EQ(sliced[1], 4);

    // Test invalid slice ranges
    EXPECT_THROW(tensor_4d_shape.slice(3, 2), std::out_of_range);
    EXPECT_THROW(tensor_4d_shape.slice(0, 5), std::out_of_range);
    EXPECT_THROW(tensor_4d_shape.slice(4, 5), std::out_of_range);
}

TEST_F(ShapeTest, Equality) {
    Shape same_matrix({5, 8});
    Shape different_matrix({8, 5});

    EXPECT_TRUE(matrix_shape == same_matrix);
    EXPECT_FALSE(matrix_shape == different_matrix);
    EXPECT_TRUE(matrix_shape != different_matrix);
    EXPECT_FALSE(matrix_shape != same_matrix);
}

TEST_F(ShapeTest, Iterators) {
    std::vector<index_t> dims_vector;
    for (auto it = tensor_3d_shape.begin(); it != tensor_3d_shape.end(); ++it) {
        dims_vector.push_back(*it);
    }
    
    EXPECT_EQ(dims_vector.size(), 3);
    EXPECT_EQ(dims_vector[0], 2);
    EXPECT_EQ(dims_vector[1], 3);
    EXPECT_EQ(dims_vector[2], 4);
}

TEST_F(ShapeTest, ToVector) {
    std::vector<index_t> vec = tensor_3d_shape.to_vector();
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 2);
    EXPECT_EQ(vec[1], 3);
    EXPECT_EQ(vec[2], 4);
}

TEST_F(ShapeTest, MakeShapeHelper) {
    Shape shape = make_shape(2, 3, 4);
    EXPECT_EQ(shape.ndims(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
}

TEST_F(ShapeTest, EdgeCases) {
    // Test maximum dimensions
    std::vector<index_t> max_dims(config::MAX_TENSOR_DIMS, 1);
    Shape max_shape(max_dims);
    EXPECT_EQ(max_shape.ndims(), config::MAX_TENSOR_DIMS);

    // Test too many dimensions (should throw)
    std::vector<index_t> too_many_dims(config::MAX_TENSOR_DIMS + 1, 1);

    
    // EXPECT_THROW seems to have issues, let's test manually
    bool exception_thrown = false;
    try {
        Shape test_shape(too_many_dims);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
        EXPECT_STREQ(e.what(), "Too many dimensions");
    }
    EXPECT_TRUE(exception_thrown) << "Expected std::invalid_argument to be thrown";
}

TEST_F(ShapeTest, LargeDimensions) {
    // Test with large dimension values
    Shape large_shape({1000000, 1000});
    EXPECT_EQ(large_shape.numel(), 1000000000LL);
    EXPECT_FALSE(large_shape.empty());
}

TEST_F(ShapeTest, Performance) {
    // Test performance with many operations
    Shape test_shape({100, 200, 300});
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        index_t numel = test_shape.numel();
        bool is_matrix = test_shape.is_matrix();
        index_t last = test_shape.last_dim();
        (void)numel; (void)is_matrix; (void)last; // Suppress unused variable warnings
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // Should complete within reasonable time (< 10ms)
    EXPECT_LT(duration.count(), 10000);
} 