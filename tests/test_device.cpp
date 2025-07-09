/**
 * @file test_device.cpp
 * @brief Unit tests for CUDA device management
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "choreo-ir/core/device.hpp"
#include "choreo-ir/core/types.hpp"
#include "choreo-ir/utils/cuda_utils.hpp"

using namespace choreo_ir;

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available, skipping device tests";
        }
    }
    
    void TearDown() override {
        // Clean up any device state
        device::finalize();
    }
};

/**
 * @brief Test device manager initialization
 */
TEST_F(DeviceTest, Initialization) {
    // Test initialization
    EXPECT_TRUE(device::initialize());
    
    // Test double initialization (should return true)
    EXPECT_TRUE(device::initialize());
    
    // Check if manager is properly initialized
    EXPECT_TRUE(device::DeviceManager::instance().is_initialized());
    
    // Test finalization
    device::finalize();
    EXPECT_FALSE(device::DeviceManager::instance().is_initialized());
}

/**
 * @brief Test device count and information
 */
TEST_F(DeviceTest, DeviceCount) {
    ASSERT_TRUE(device::initialize());
    
    // Should have at least one device
    EXPECT_GT(device::get_device_count(), 0);
    
    // Test device information access
    int current_device = device::get_current_device();
    EXPECT_GE(current_device, 0);
    
    const auto& info = device::get_device_info(current_device);
    EXPECT_EQ(info.device_id, current_device);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.total_memory, 0);
    EXPECT_GT(info.multiprocessor_count, 0);
    EXPECT_GT(info.max_threads_per_block, 0);
    EXPECT_EQ(info.warp_size, 32);
}

/**
 * @brief Test device switching
 */
TEST_F(DeviceTest, DeviceSwitching) {
    ASSERT_TRUE(device::initialize());
    
    int device_count = device::get_device_count();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least 2 devices for switching test";
    }
    
    // Get all available device IDs
    std::vector<int> device_ids;
    for (int i = 0; i < device_count; ++i) {
        try {
            const auto& info = device::get_device_info(i);
            device_ids.push_back(info.device_id);
        } catch (const std::exception&) {
            // Device not in supported list, skip
        }
    }
    
    if (device_ids.size() < 2) {
        GTEST_SKIP() << "Need at least 2 supported devices for switching test";
    }
    
    // Test switching between devices
    int original_device = device::get_current_device();
    
    for (int device_id : device_ids) {
        EXPECT_TRUE(device::set_device(device_id));
        EXPECT_EQ(device::get_current_device(), device_id);
    }
    
    // Test invalid device ID
    EXPECT_FALSE(device::set_device(999));
    
    // Restore original device
    EXPECT_TRUE(device::set_device(original_device));
}

/**
 * @brief Test compute capability detection
 */
TEST_F(DeviceTest, ComputeCapability) {
    ASSERT_TRUE(device::initialize());
    
    ComputeCapability capability = device::get_compute_capability();
    
    // Should be at least SM 7.0 (our minimum requirement)
    EXPECT_GE(static_cast<int>(capability), static_cast<int>(ComputeCapability::SM_70));
    
    // Test device-specific capability
    int current_device = device::get_current_device();
    const auto& info = device::get_device_info(current_device);
    
    EXPECT_EQ(capability, info.compute_capability);
}

/**
 * @brief Test tensor core support detection
 */
TEST_F(DeviceTest, TensorCoreSupport) {
    ASSERT_TRUE(device::initialize());
    
    bool tensor_core_support = device::is_tensor_core_supported();
    
    // Should support tensor cores for SM 7.0+
    ComputeCapability capability = device::get_compute_capability();
    bool expected_support = static_cast<int>(capability) >= 70;
    
    EXPECT_EQ(tensor_core_support, expected_support);
    
    // Check consistency with device info
    int current_device = device::get_current_device();
    const auto& info = device::get_device_info(current_device);
    EXPECT_EQ(tensor_core_support, info.tensor_core_support);
}

/**
 * @brief Test device synchronization
 */
TEST_F(DeviceTest, Synchronization) {
    ASSERT_TRUE(device::initialize());
    
    // Test synchronization (should not crash)
    device::synchronize();
    
    // Launch a simple kernel to test synchronization
    auto stream = cuda_utils::create_stream();
    
    // Allocate some memory
    void* ptr = cuda_utils::malloc_device(1024);
    
    // Set memory asynchronously
    cuda_utils::memset_async(ptr, 0, 1024, stream);
    
    // Synchronize and check
    cuda_utils::synchronize_stream(stream);
    device::synchronize();
    
    // Cleanup
    cuda_utils::free_device(ptr);
    cuda_utils::destroy_stream(stream);
}

/**
 * @brief Test memory information
 */
TEST_F(DeviceTest, MemoryInfo) {
    ASSERT_TRUE(device::initialize());
    
    const auto& info = device::get_device_info();
    
    EXPECT_GT(info.total_memory, 0);
    EXPECT_GT(info.free_memory, 0);
    EXPECT_LE(info.free_memory, info.total_memory);
    
    // Test consistency with CUDA API
    size_t free_mem, total_mem;
    cuda_utils::get_memory_info(free_mem, total_mem);
    
    EXPECT_EQ(info.total_memory, total_mem);
    // free_memory may differ slightly due to timing
}

/**
 * @brief Test device properties validation
 */
TEST_F(DeviceTest, DeviceProperties) {
    ASSERT_TRUE(device::initialize());
    
    const auto& info = device::get_device_info();
    
    // Validate reasonable property values
    EXPECT_GT(info.multiprocessor_count, 0);
    EXPECT_GE(info.multiprocessor_count, 1);
    EXPECT_LE(info.multiprocessor_count, 128); // Reasonable upper bound
    
    EXPECT_GT(info.max_threads_per_block, 0);
    EXPECT_GE(info.max_threads_per_block, 512); // Minimum for modern GPUs
    EXPECT_LE(info.max_threads_per_block, 2048); // Maximum allowed
    
    EXPECT_GT(info.max_shared_memory_per_block, 0);
    EXPECT_GE(info.max_shared_memory_per_block, 16384); // 16KB minimum
    
    EXPECT_EQ(info.warp_size, 32); // Always 32 for NVIDIA GPUs
}

/**
 * @brief Test error handling
 */
TEST_F(DeviceTest, ErrorHandling) {
    ASSERT_TRUE(device::initialize());
    
    // Test accessing non-existent device
    EXPECT_THROW(device::get_device_info(999), std::runtime_error);
    
    // Test setting invalid device
    EXPECT_FALSE(device::set_device(-1));
    EXPECT_FALSE(device::set_device(999));
    
    // Current device should remain unchanged
    int original_device = device::get_current_device();
    device::set_device(999);
    EXPECT_EQ(device::get_current_device(), original_device);
}

/**
 * @brief Test singleton pattern
 */
TEST_F(DeviceTest, SingletonPattern) {
    // Multiple instances should be the same
    auto& manager1 = device::DeviceManager::instance();
    auto& manager2 = device::DeviceManager::instance();
    
    EXPECT_EQ(&manager1, &manager2);
    
    // Test thread safety (basic check)
    ASSERT_TRUE(device::initialize());
    
    bool success1 = device::initialize();
    bool success2 = device::initialize();
    
    EXPECT_TRUE(success1);
    EXPECT_TRUE(success2);
}

/**
 * @brief Performance test - device operations should be fast
 */
TEST_F(DeviceTest, Performance) {
    ASSERT_TRUE(device::initialize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform multiple device queries
    for (int i = 0; i < 1000; ++i) {
        device::get_current_device();
        device::is_tensor_core_supported();
        device::get_compute_capability();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete within reasonable time (less than 1ms total)
    EXPECT_LT(duration.count(), 1000);
}

/**
 * @brief Integration test with CUDA runtime
 */
TEST_F(DeviceTest, CudaRuntimeIntegration) {
    ASSERT_TRUE(device::initialize());
    
    int current_device = device::get_current_device();
    
    // Check consistency with CUDA runtime
    int cuda_device;
    ASSERT_EQ(cudaGetDevice(&cuda_device), cudaSuccess);
    EXPECT_EQ(current_device, cuda_device);
    
    // Test device switching consistency
    if (device::get_device_count() > 1) {
        // Find another device
        for (int i = 0; i < device::get_device_count(); ++i) {
            try {
                const auto& info = device::get_device_info(i);
                if (info.device_id != current_device) {
                    ASSERT_TRUE(device::set_device(info.device_id));
                    
                    ASSERT_EQ(cudaGetDevice(&cuda_device), cudaSuccess);
                    EXPECT_EQ(info.device_id, cuda_device);
                    
                    // Restore original device
                    ASSERT_TRUE(device::set_device(current_device));
                    break;
                }
            } catch (const std::exception&) {
                // Skip unsupported devices
            }
        }
    }
} 