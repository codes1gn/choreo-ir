/**
 * @file test_cuda_utils.cpp
 * @brief Unit tests for CUDA utilities
 */

#include <gtest/gtest.h>
#include <vector>
#include "choreo-ir/utils/cuda_utils.hpp"
#include "choreo-ir/core/device.hpp"

using namespace choreo_ir;

class CudaUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize device subsystem
        if (!device::initialize()) {
            GTEST_SKIP() << "No CUDA devices available, skipping CUDA utils tests";
        }
    }

    void TearDown() override {
        device::finalize();
    }
};

// Test error checking macros (should not throw for successful operations)
TEST_F(CudaUtilsTest, ErrorChecking) {
    // Test CUDA_CHECK with successful operation
    EXPECT_NO_THROW({
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
    });
    
    // Test getting device properties
    EXPECT_NO_THROW({
        auto props = cuda_utils::get_device_properties(0);
        EXPECT_GT(props.totalGlobalMem, 0);
    });
}

// Test memory allocation and deallocation
TEST_F(CudaUtilsTest, MemoryManagement) {
    const size_t size = 1024 * sizeof(float);
    
    // Test device memory allocation
    void* device_ptr = nullptr;
    EXPECT_NO_THROW({
        device_ptr = cuda_utils::malloc_device(size);
    });
    EXPECT_NE(device_ptr, nullptr);
    
    // Test device memory deallocation
    EXPECT_NO_THROW({
        cuda_utils::free_device(device_ptr);
    });
    
    // Test host memory allocation
    void* host_ptr = nullptr;
    EXPECT_NO_THROW({
        host_ptr = cuda_utils::malloc_host(size);
    });
    EXPECT_NE(host_ptr, nullptr);
    
    // Test host memory deallocation
    EXPECT_NO_THROW({
        cuda_utils::free_host(host_ptr);
    });
}

// Test memory copy operations
TEST_F(CudaUtilsTest, MemoryCopy) {
    const size_t size = 1024;
    const size_t bytes = size * sizeof(float);
    
    // Allocate host and device memory
    float* host_src = static_cast<float*>(cuda_utils::malloc_host(bytes));
    float* host_dst = static_cast<float*>(cuda_utils::malloc_host(bytes));
    float* device_ptr = static_cast<float*>(cuda_utils::malloc_device(bytes));
    
    // Initialize host source data
    for (size_t i = 0; i < size; ++i) {
        host_src[i] = static_cast<float>(i);
    }
    
    // Test host to device copy
    EXPECT_NO_THROW({
        cuda_utils::memcpy(device_ptr, host_src, bytes, cudaMemcpyHostToDevice);
    });
    
    // Test device to host copy
    EXPECT_NO_THROW({
        cuda_utils::memcpy(host_dst, device_ptr, bytes, cudaMemcpyDeviceToHost);
    });
    
    // Verify data integrity
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(host_src[i], host_dst[i]);
    }
    
    // Cleanup
    cuda_utils::free_host(host_src);
    cuda_utils::free_host(host_dst);
    cuda_utils::free_device(device_ptr);
}

// Test memory set operations
TEST_F(CudaUtilsTest, MemorySet) {
    const size_t size = 1024;
    const size_t bytes = size * sizeof(int);
    
    // Allocate device memory
    int* device_ptr = static_cast<int*>(cuda_utils::malloc_device(bytes));
    int* host_ptr = static_cast<int*>(cuda_utils::malloc_host(bytes));
    
    // Set device memory to zero
    EXPECT_NO_THROW({
        cuda_utils::memset(device_ptr, 0, bytes);
    });
    
    // Copy back to host and verify
    cuda_utils::memcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host_ptr[i], 0);
    }
    
    // Cleanup
    cuda_utils::free_device(device_ptr);
    cuda_utils::free_host(host_ptr);
}

// Test device utilities
TEST_F(CudaUtilsTest, DeviceUtilities) {
    // Test getting current device
    int device_id;
    EXPECT_NO_THROW({
        device_id = cuda_utils::get_current_device();
    });
    EXPECT_GE(device_id, 0);
    
    // Test device synchronization
    EXPECT_NO_THROW({
        cuda_utils::synchronize();
    });
    
    // Test memory info
    size_t free_mem, total_mem;
    EXPECT_NO_THROW({
        cuda_utils::get_memory_info(free_mem, total_mem);
    });
    EXPECT_GT(total_mem, 0);
    EXPECT_LE(free_mem, total_mem);
    
    // Test compute capability detection
    ComputeCapability capability;
    EXPECT_NO_THROW({
        capability = cuda_utils::get_compute_capability(device_id);
    });
    
    // Test tensor core support detection
    bool tensor_core_support;
    EXPECT_NO_THROW({
        tensor_core_support = cuda_utils::supports_tensor_core(device_id);
    });
}

// Test RAII stream wrapper
TEST_F(CudaUtilsTest, StreamWrapper) {
    // Test stream creation and destruction
    cuda_utils::CudaStream stream;
    EXPECT_NE(stream.get(), nullptr);
    
    // Test stream synchronization
    EXPECT_NO_THROW({
        stream.synchronize();
    });
    
    // Test move semantics
    cuda_utils::CudaStream moved_stream = std::move(stream);
    EXPECT_NE(moved_stream.get(), nullptr);
    EXPECT_EQ(stream.get(), nullptr);  // Original should be null after move
}

// Test RAII event wrapper
TEST_F(CudaUtilsTest, EventWrapper) {
    // Test event creation and destruction
    cuda_utils::CudaEvent event;
    EXPECT_NE(event.get(), nullptr);
    
    // Test event recording and waiting
    EXPECT_NO_THROW({
        event.record();
        event.wait();
    });
    
    // Test move semantics
    cuda_utils::CudaEvent moved_event = std::move(event);
    EXPECT_NE(moved_event.get(), nullptr);
    EXPECT_EQ(event.get(), nullptr);  // Original should be null after move
}

// Test asynchronous operations
TEST_F(CudaUtilsTest, AsynchronousOperations) {
    const size_t size = 1024;
    const size_t bytes = size * sizeof(float);
    
    // Create stream
    cuda_utils::CudaStream stream;
    
    // Allocate memory
    float* host_ptr = static_cast<float*>(cuda_utils::malloc_host(bytes));
    float* device_ptr = static_cast<float*>(cuda_utils::malloc_device(bytes));
    
    // Initialize host data
    for (size_t i = 0; i < size; ++i) {
        host_ptr[i] = static_cast<float>(i);
    }
    
    // Test async memory copy
    EXPECT_NO_THROW({
        cuda_utils::memcpy_async(device_ptr, host_ptr, bytes, 
                                cudaMemcpyHostToDevice, stream);
    });
    
    // Test async memory set
    EXPECT_NO_THROW({
        cuda_utils::memset_async(device_ptr, 0, bytes, stream);
    });
    
    // Synchronize stream
    stream.synchronize();
    
    // Cleanup
    cuda_utils::free_host(host_ptr);
    cuda_utils::free_device(device_ptr);
}

// Test timing with events
TEST_F(CudaUtilsTest, Timing) {
    cuda_utils::CudaEvent start_event, end_event;
    
    // Record start event
    start_event.record();
    
    // Do some work (dummy computation)
    const size_t size = 1024 * 1024;
    float* device_ptr = static_cast<float*>(cuda_utils::malloc_device(size * sizeof(float)));
    cuda_utils::memset(device_ptr, 0, size * sizeof(float));
    
    // Record end event
    end_event.record();
    end_event.wait();
    
    // Get elapsed time
    float elapsed_time = start_event.elapsed_time(end_event);
    EXPECT_GE(elapsed_time, 0.0f);
    
    // Cleanup
    cuda_utils::free_device(device_ptr);
}

// Test managed memory (if supported)
TEST_F(CudaUtilsTest, ManagedMemory) {
    const size_t size = 1024;
    const size_t bytes = size * sizeof(float);
    
    // Try to allocate managed memory
    void* managed_ptr = nullptr;
    EXPECT_NO_THROW({
        managed_ptr = cuda_utils::malloc_managed(bytes);
    });
    EXPECT_NE(managed_ptr, nullptr);
    
    // Test accessing from host (managed memory should be accessible)
    float* float_ptr = static_cast<float*>(managed_ptr);
    for (size_t i = 0; i < size; ++i) {
        float_ptr[i] = static_cast<float>(i);
    }
    
    // Synchronize to ensure device operations complete
    cuda_utils::synchronize();
    
    // Verify data (should be accessible from host)
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(float_ptr[i], static_cast<float>(i));
    }
    
    // Free managed memory using device free (works for managed memory too)
    cuda_utils::free_device(managed_ptr);
}

// Test error handling with invalid operations
TEST_F(CudaUtilsTest, ErrorHandlingDisabled) {
    // This test assumes error checking can be temporarily disabled
    // In practice, we'd need a way to toggle config::ENABLE_CUDA_ERROR_CHECKING
    
    // For now, just test that normal operations work
    void* ptr = cuda_utils::malloc_device(1024);
    EXPECT_NE(ptr, nullptr);
    cuda_utils::free_device(ptr);
} 