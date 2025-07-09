/**
 * @file test_tensor_simple.cpp
 * @brief Simplified unit tests for Tensor class (host functionality only)
 */

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <chrono>
#include "choreo-ir/tensor/shape.hpp"
#include "choreo-ir/tensor/stride.hpp"
#include "choreo-ir/tensor/layout.hpp"
#include "choreo-ir/core/types.hpp"

// Simple Tensor class for testing (host-only)
namespace choreo_ir {
template<typename T>
class SimpleTensor {
public:
    SimpleTensor() : data_(nullptr), layout_(), memory_type_(MemoryType::HOST) {}
    
    explicit SimpleTensor(const Shape& shape, MemoryType memory_type = MemoryType::HOST)
        : layout_(shape), memory_type_(memory_type) {
        allocate();
    }
    
    SimpleTensor(const SimpleTensor<T>& other) : layout_(other.layout_), memory_type_(other.memory_type_) {
        allocate();
        copy_from(other);
    }
    
    SimpleTensor(SimpleTensor<T>&& other) noexcept 
        : data_(other.data_), layout_(std::move(other.layout_)), memory_type_(other.memory_type_) {
        other.data_ = nullptr;
    }
    
    ~SimpleTensor() { deallocate(); }
    
    SimpleTensor<T>& operator=(const SimpleTensor<T>& other) {
        if (this != &other) {
            deallocate();
            layout_ = other.layout_;
            memory_type_ = other.memory_type_;
            allocate();
            copy_from(other);
        }
        return *this;
    }
    
    SimpleTensor<T>& operator=(SimpleTensor<T>&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            layout_ = std::move(other.layout_);
            memory_type_ = other.memory_type_;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    // Basic properties
    T* data() const { return data_; }
    const Layout& layout() const { return layout_; }
    const Shape& shape() const { return layout_.shape(); }
    dim_t ndims() const { return layout_.ndims(); }
    index_t numel() const { return layout_.numel(); }
    DataType dtype() const { return get_dtype<T>(); }
    MemoryType memory_type() const { return memory_type_; }
    bool empty() const { return data_ == nullptr || layout_.numel() == 0; }
    bool is_contiguous() const { return layout_.is_contiguous(); }
    
    // Element access
    template<typename... Args>
    T& operator()(Args... indices) const {
        return data_[layout_.offset(indices...)];
    }
    
    // Operations
    void fill(T value) {
        if (data_ == nullptr || layout_.numel() == 0) {
            return;
        }
        std::fill_n(data_, layout_.numel(), value);
    }
    
    void copy_from_host(const T* host_data) {
        if (data_ == nullptr || host_data == nullptr || layout_.numel() == 0) {
            return;
        }
        std::memcpy(data_, host_data, layout_.numel() * sizeof(T));
    }
    
    void copy_to_host(T* host_data) const {
        if (data_ == nullptr || host_data == nullptr || layout_.numel() == 0) {
            return;
        }
        std::memcpy(host_data, data_, layout_.numel() * sizeof(T));
    }
    
    // Static factory methods
    static SimpleTensor<T> zeros(const Shape& shape) {
        SimpleTensor<T> tensor(shape);
        tensor.fill(T(0));
        return tensor;
    }
    
    static SimpleTensor<T> ones(const Shape& shape) {
        SimpleTensor<T> tensor(shape);
        tensor.fill(T(1));
        return tensor;
    }

private:
    T* data_;
    Layout layout_;
    MemoryType memory_type_;
    
    void allocate() {
        if (layout_.numel() == 0) {
            data_ = nullptr;
            return;
        }
        data_ = new T[layout_.numel()];
    }
    
    void deallocate() {
        if (data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
        }
    }
    
    void copy_from(const SimpleTensor<T>& other) {
        if (data_ == nullptr || other.data_ == nullptr || 
            layout_.numel() != other.layout_.numel()) {
            return;
        }
        std::memcpy(data_, other.data_, layout_.numel() * sizeof(T));
    }
};
}

using namespace choreo_ir;

class SimpleTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        shape_2d_ = Shape({4, 6});
        shape_3d_ = Shape({2, 3, 4});
        shape_1d_ = Shape({24});
    }

    Shape shape_2d_;
    Shape shape_3d_;
    Shape shape_1d_;
};

// Test constructors
TEST_F(SimpleTensorTest, DefaultConstructor) {
    SimpleTensor<float> tensor;
    
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_EQ(tensor.memory_type(), MemoryType::HOST);
    EXPECT_EQ(tensor.numel(), 1);  // Default shape has numel = 1
}

TEST_F(SimpleTensorTest, ConstructorFromShape) {
    SimpleTensor<float> tensor(shape_2d_);
    
    EXPECT_FALSE(tensor.empty());
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.shape(), shape_2d_);
    EXPECT_EQ(tensor.memory_type(), MemoryType::HOST);
    EXPECT_EQ(tensor.numel(), 24);
    EXPECT_TRUE(tensor.is_contiguous());
}

// Test copy and move constructors
TEST_F(SimpleTensorTest, CopyConstructor) {
    SimpleTensor<float> original(shape_2d_);
    original.fill(3.14f);
    
    SimpleTensor<float> copy(original);
    
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

TEST_F(SimpleTensorTest, MoveConstructor) {
    SimpleTensor<float> original(shape_2d_);
    float* original_data_ptr = original.data();
    
    SimpleTensor<float> moved(std::move(original));
    
    EXPECT_EQ(moved.data(), original_data_ptr);
    EXPECT_EQ(original.data(), nullptr);  // Original should be empty
    EXPECT_EQ(moved.shape(), shape_2d_);
}

// Test assignment operators
TEST_F(SimpleTensorTest, CopyAssignment) {
    SimpleTensor<float> tensor1(shape_2d_);
    SimpleTensor<float> tensor2(shape_3d_);
    
    tensor1.fill(1.0f);
    tensor2.fill(2.0f);
    
    tensor2 = tensor1;
    
    EXPECT_EQ(tensor2.shape(), tensor1.shape());
    EXPECT_NE(tensor2.data(), tensor1.data());
}

TEST_F(SimpleTensorTest, MoveAssignment) {
    SimpleTensor<float> tensor1(shape_2d_);
    SimpleTensor<float> tensor2(shape_3d_);
    
    float* tensor1_data = tensor1.data();
    tensor2 = std::move(tensor1);
    
    EXPECT_EQ(tensor2.data(), tensor1_data);
    EXPECT_EQ(tensor1.data(), nullptr);
}

// Test basic properties
TEST_F(SimpleTensorTest, BasicProperties) {
    SimpleTensor<float> tensor(shape_3d_);
    
    EXPECT_EQ(tensor.ndims(), 3);
    EXPECT_EQ(tensor.numel(), 24);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.memory_type(), MemoryType::HOST);
    EXPECT_FALSE(tensor.empty());
    EXPECT_TRUE(tensor.is_contiguous());
}

// Test data operations
TEST_F(SimpleTensorTest, FillOperation) {
    SimpleTensor<float> tensor(shape_2d_);
    
    tensor.fill(5.5f);
    
    std::vector<float> data(tensor.numel());
    tensor.copy_to_host(data.data());
    
    for (float value : data) {
        EXPECT_FLOAT_EQ(value, 5.5f);
    }
}

TEST_F(SimpleTensorTest, HostCopy) {
    SimpleTensor<float> tensor(shape_2d_);
    
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
TEST_F(SimpleTensorTest, ElementAccess) {
    SimpleTensor<float> tensor(shape_2d_);
    tensor.fill(0.0f);
    
    // Set some values
    tensor(1, 2) = 42.0f;
    tensor(0, 0) = 1.0f;
    
    EXPECT_FLOAT_EQ(tensor(1, 2), 42.0f);
    EXPECT_FLOAT_EQ(tensor(0, 0), 1.0f);
}

// Test static factory methods
TEST_F(SimpleTensorTest, FactoryMethods) {
    auto zeros_tensor = SimpleTensor<float>::zeros(shape_2d_);
    auto ones_tensor = SimpleTensor<float>::ones(shape_2d_);
    
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
TEST_F(SimpleTensorTest, DifferentDataTypes) {
    SimpleTensor<int32_t> int_tensor(shape_2d_);
    SimpleTensor<double> double_tensor(shape_2d_);
    
    EXPECT_EQ(int_tensor.dtype(), DataType::INT32);
    EXPECT_EQ(double_tensor.dtype(), DataType::FLOAT64);
    
    int_tensor.fill(42);
    double_tensor.fill(3.14159);
    
    EXPECT_EQ(int_tensor(0, 0), 42);
    EXPECT_DOUBLE_EQ(double_tensor(0, 0), 3.14159);
}

// Test edge cases
TEST_F(SimpleTensorTest, EdgeCases) {
    // Empty shape
    Shape empty_shape({0, 5});
    SimpleTensor<float> empty_tensor(empty_shape);
    EXPECT_TRUE(empty_tensor.empty());
    
    // Single element tensor
    Shape single_shape({1});
    SimpleTensor<float> single_tensor(single_shape);
    EXPECT_EQ(single_tensor.numel(), 1);
    EXPECT_FALSE(single_tensor.empty());
    
    // Large tensor (within reasonable limits)
    Shape large_shape({100, 100});
    SimpleTensor<float> large_tensor(large_shape);
    EXPECT_EQ(large_tensor.numel(), 10000);
}

// Performance test
TEST_F(SimpleTensorTest, PerformanceTest) {
    Shape large_shape({1000, 1000});
    SimpleTensor<float> tensor(large_shape);
    
    // Time fill operation
    auto start = std::chrono::high_resolution_clock::now();
    tensor.fill(1.0f);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time (less than 1 second for 1M elements)
    EXPECT_LT(duration.count(), 1000);
} 