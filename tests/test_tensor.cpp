/**
 * @file test_tensor.cpp
 * @brief Comprehensive unit tests for Tensor class
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <chrono>
#include "choreo-ir/tensor/tensor.hpp"

using namespace choreo_ir;

class TensorTest : public ::testing::Test {
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
TEST_F(TensorTest, DefaultConstructor) {
    Tensor<float> tensor;
    
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.memory_type(), MemoryType::DEVICE);
    EXPECT_EQ(tensor.numel(), 1);  // Default shape has numel = 1
}

TEST_F(TensorTest, ConstructorFromShape) {
    Tensor<float> host_tensor(shape_2d_, MemoryType::HOST);
    
    EXPECT_FALSE(host_tensor.empty());
    EXPECT_NE(host_tensor.data(), nullptr);
    EXPECT_EQ(host_tensor.shape(), shape_2d_);
    EXPECT_EQ(host_tensor.memory_type(), MemoryType::HOST);
    EXPECT_EQ(host_tensor.numel(), 24);
    EXPECT_TRUE(host_tensor.is_contiguous());
}

TEST_F(TensorTest, ConstructorFromLayout) {
    Layout layout(shape_2d_, LayoutType::COLUMN_MAJOR);
    Tensor<float> tensor(layout, MemoryType::HOST);
    
    EXPECT_EQ(tensor.layout(), layout);
    EXPECT_EQ(tensor.shape(), shape_2d_);
    EXPECT_EQ(tensor.memory_type(), MemoryType::HOST);
}

// Test copy and move constructors
TEST_F(TensorTest, CopyConstructor) {
    Tensor<float> original(shape_2d_, MemoryType::HOST);
    original.fill(3.14f);
    
    Tensor<float> copy(original);
    
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.memory_type(), original.memory_type());
    EXPECT_NE(copy.data(), original.data());  // Different memory
    
    // Verify data was copied
    std::vector<float> original_data(original.numel());
    std::vector<float> copy_data(copy.numel());
    original.copy_to_host(original_data.data());
    copy.copy_to_host(copy_data.data());
    
    for (size_t i = 0; i < original_data.size(); ++i) {
        EXPECT_FLOAT_EQ(original_data[i], copy_data[i]);
    }
}

TEST_F(TensorTest, MoveConstructor) {
    Tensor<float> original(shape_2d_, MemoryType::HOST);
    float* original_data_ptr = original.data();
    
    Tensor<float> moved(std::move(original));
    
    EXPECT_EQ(moved.data(), original_data_ptr);
    EXPECT_EQ(original.data(), nullptr);  // Original should be empty
    EXPECT_EQ(moved.shape(), shape_2d_);
}

// Test assignment operators
TEST_F(TensorTest, CopyAssignment) {
    Tensor<float> tensor1(shape_2d_, MemoryType::HOST);
    Tensor<float> tensor2(shape_3d_, MemoryType::HOST);
    
    tensor1.fill(1.0f);
    tensor2.fill(2.0f);
    
    tensor2 = tensor1;
    
    EXPECT_EQ(tensor2.shape(), tensor1.shape());
    EXPECT_NE(tensor2.data(), tensor1.data());
}

TEST_F(TensorTest, MoveAssignment) {
    Tensor<float> tensor1(shape_2d_, MemoryType::HOST);
    Tensor<float> tensor2(shape_3d_, MemoryType::HOST);
    
    float* tensor1_data = tensor1.data();
    tensor2 = std::move(tensor1);
    
    EXPECT_EQ(tensor2.data(), tensor1_data);
    EXPECT_EQ(tensor1.data(), nullptr);
}

// Test basic properties
TEST_F(TensorTest, BasicProperties) {
    Tensor<float> tensor(shape_3d_, MemoryType::HOST);
    
    EXPECT_EQ(tensor.ndims(), 3);
    EXPECT_EQ(tensor.numel(), 24);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.memory_type(), MemoryType::HOST);
    EXPECT_FALSE(tensor.empty());
    EXPECT_TRUE(tensor.is_contiguous());
}

// Test data operations
TEST_F(TensorTest, FillOperation) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    
    tensor.fill(5.5f);
    
    std::vector<float> data(tensor.numel());
    tensor.copy_to_host(data.data());
    
    for (float value : data) {
        EXPECT_FLOAT_EQ(value, 5.5f);
    }
}

TEST_F(TensorTest, HostDeviceCopy) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    
    // Fill with test data
    std::vector<float> test_data(tensor.numel());
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i);
    }
    
    tensor.copy_from_host(test_data.data());
    
    // Verify data was copied correctly
    std::vector<float> result_data(tensor.numel());
    tensor.copy_to_host(result_data.data());
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result_data[i], test_data[i]);
    }
}

// Test element access
TEST_F(TensorTest, ElementAccess) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    tensor.fill(0.0f);
    
    // Set some values
    tensor(1, 2) = 42.0f;
    tensor(0, 0) = 1.0f;
    
    EXPECT_FLOAT_EQ(tensor(1, 2), 42.0f);
    EXPECT_FLOAT_EQ(tensor(0, 0), 1.0f);
}

// Test views
TEST_F(TensorTest, ViewCreation) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    tensor.fill(7.0f);
    
    auto view = tensor.view();
    
    EXPECT_EQ(view.data(), tensor.data());
    EXPECT_EQ(view.shape(), tensor.shape());
    EXPECT_EQ(view.memory_space(), MemorySpace::GLOBAL);
}

TEST_F(TensorTest, SliceView) {
    Tensor<float> tensor(shape_3d_, MemoryType::HOST);
    
    auto slice = tensor.slice(1, 3);
    
    EXPECT_EQ(slice.ndims(), 2);
    EXPECT_EQ(slice.shape()[0], shape_3d_[1]);
    EXPECT_EQ(slice.shape()[1], shape_3d_[2]);
}

TEST_F(TensorTest, TransposeView) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    
    auto transposed = tensor.transpose();
    
    EXPECT_EQ(transposed.shape()[0], shape_2d_[1]);
    EXPECT_EQ(transposed.shape()[1], shape_2d_[0]);
}

TEST_F(TensorTest, ReshapeView) {
    Tensor<float> tensor(shape_1d_, MemoryType::HOST);
    
    auto reshaped = tensor.reshape(shape_2d_);
    
    EXPECT_EQ(reshaped.shape(), shape_2d_);
    EXPECT_EQ(reshaped.numel(), tensor.numel());
}

// Test resize operation
TEST_F(TensorTest, ResizeOperation) {
    Tensor<float> tensor(shape_2d_, MemoryType::HOST);
    float* original_data = tensor.data();
    
    // Resize to same size - should keep same data
    tensor.resize(shape_2d_);
    EXPECT_EQ(tensor.data(), original_data);
    
    // Resize to different size - should reallocate
    tensor.resize(shape_3d_);
    EXPECT_NE(tensor.data(), original_data);
    EXPECT_EQ(tensor.shape(), shape_3d_);
}

// Test static factory methods
TEST_F(TensorTest, FactoryMethods) {
    auto host_tensor = Tensor<float>::host(shape_2d_);
    auto device_tensor = Tensor<float>::device(shape_2d_);
    auto zeros_tensor = Tensor<float>::zeros(shape_2d_, MemoryType::HOST);
    auto ones_tensor = Tensor<float>::ones(shape_2d_, MemoryType::HOST);
    
    EXPECT_EQ(host_tensor.memory_type(), MemoryType::HOST);
    EXPECT_EQ(device_tensor.memory_type(), MemoryType::DEVICE);
    
    // Verify zeros
    std::vector<float> zeros_data(zeros_tensor.numel());
    zeros_tensor.copy_to_host(zeros_data.data());
    for (float value : zeros_data) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
    
    // Verify ones
    std::vector<float> ones_data(ones_tensor.numel());
    ones_tensor.copy_to_host(ones_data.data());
    for (float value : ones_data) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }
}

// Test different data types
TEST_F(TensorTest, DifferentDataTypes) {
    Tensor<int32_t> int_tensor(shape_2d_, MemoryType::HOST);
    Tensor<double> double_tensor(shape_2d_, MemoryType::HOST);
    
    EXPECT_EQ(int_tensor.dtype(), DataType::INT32);
    EXPECT_EQ(double_tensor.dtype(), DataType::FLOAT64);
    
    int_tensor.fill(42);
    double_tensor.fill(3.14159);
    
    EXPECT_EQ(int_tensor(0, 0), 42);
    EXPECT_DOUBLE_EQ(double_tensor(0, 0), 3.14159);
}

// Test memory space mapping
TEST_F(TensorTest, MemorySpaceMapping) {
    Tensor<float> host_tensor(shape_2d_, MemoryType::HOST);
    Tensor<float> device_tensor(shape_2d_, MemoryType::DEVICE);
    
    auto host_view = host_tensor.view();
    auto device_view = device_tensor.view();
    
    EXPECT_EQ(host_view.memory_space(), MemorySpace::GLOBAL);
    EXPECT_EQ(device_view.memory_space(), MemorySpace::GLOBAL);
}

// Test edge cases
TEST_F(TensorTest, EdgeCases) {
    // Empty shape
    Shape empty_shape({0, 5});
    Tensor<float> empty_tensor(empty_shape, MemoryType::HOST);
    EXPECT_TRUE(empty_tensor.empty());
    
    // Single element tensor
    Shape single_shape({1});
    Tensor<float> single_tensor(single_shape, MemoryType::HOST);
    EXPECT_EQ(single_tensor.numel(), 1);
    EXPECT_FALSE(single_tensor.empty());
    
    // Large tensor (within reasonable limits)
    Shape large_shape({100, 100});
    Tensor<float> large_tensor(large_shape, MemoryType::HOST);
    EXPECT_EQ(large_tensor.numel(), 10000);
}

// Performance test
TEST_F(TensorTest, PerformanceTest) {
    Shape large_shape({1000, 1000});
    Tensor<float> tensor(large_shape, MemoryType::HOST);
    
    // Time fill operation
    auto start = std::chrono::high_resolution_clock::now();
    tensor.fill(1.0f);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time (less than 1 second for 1M elements)
    EXPECT_LT(duration.count(), 1000);
} 