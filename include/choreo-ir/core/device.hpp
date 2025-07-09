/**
 * @file device.hpp
 * @brief CUDA device management and initialization
 */

#ifndef CHOREO_IR_CORE_DEVICE_HPP
#define CHOREO_IR_CORE_DEVICE_HPP

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "types.hpp"
#include "config.hpp"

namespace choreo_ir {
namespace device {

/**
 * @struct DeviceInfo
 * @brief Information about a CUDA device
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    ComputeCapability compute_capability;
    size_t total_memory;
    size_t free_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int warp_size;
    bool tensor_core_support;
    
    DeviceInfo() : device_id(-1), compute_capability(ComputeCapability::SM_70),
                   total_memory(0), free_memory(0), multiprocessor_count(0),
                   max_threads_per_block(0), max_shared_memory_per_block(0),
                   warp_size(32), tensor_core_support(false) {}
};

/**
 * @class DeviceManager
 * @brief Manages CUDA devices and contexts
 */
class DeviceManager {
public:
    /**
     * @brief Get the singleton instance
     */
    static DeviceManager& instance() {
        static DeviceManager instance_;
        return instance_;
    }

    /**
     * @brief Initialize the device manager
     * @return true if successful, false otherwise
     */
    bool initialize() {
        if (initialized_) return true;
        
        try {
            // Get device count
            cudaError_t error = cudaGetDeviceCount(&device_count_);
            if (error != cudaSuccess) {
                return false;
            }
            
            if (device_count_ == 0) {
                return false;
            }
            
            // Query information for all devices
            device_infos_.reserve(device_count_);
            for (int i = 0; i < device_count_; ++i) {
                if (!is_device_supported(i)) {
                    continue;
                }
                device_infos_.push_back(query_device_info(i));
            }
            
            if (device_infos_.empty()) {
                return false;
            }
            
            // Set the first supported device as current
            current_device_ = device_infos_[0].device_id;
            cudaSetDevice(current_device_);
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception&) {
            return false;
        }
    }

    /**
     * @brief Finalize the device manager
     */
    void finalize() {
        if (initialized_) {
            cudaDeviceReset();
            initialized_ = false;
        }
    }

    /**
     * @brief Get the number of available devices
     * @return Number of devices
     */
    int get_device_count() const {
        return static_cast<int>(device_infos_.size());
    }

    /**
     * @brief Get device information
     * @param device_id Device ID
     * @return Device information
     */
    const DeviceInfo& get_device_info(int device_id) const {
        for (const auto& info : device_infos_) {
            if (info.device_id == device_id) {
                return info;
            }
        }
        throw std::runtime_error("Device not found: " + std::to_string(device_id));
    }

    /**
     * @brief Set the current device
     * @param device_id Device ID
     * @return true if successful, false otherwise
     */
    bool set_device(int device_id) {
        if (!initialized_) return false;
        
        // Check if device exists in our supported list
        bool found = false;
        for (const auto& info : device_infos_) {
            if (info.device_id == device_id) {
                found = true;
                break;
            }
        }
        
        if (!found) return false;
        
        cudaError_t error = cudaSetDevice(device_id);
        if (error == cudaSuccess) {
            current_device_ = device_id;
            return true;
        }
        return false;
    }

    /**
     * @brief Get the current device ID
     * @return Current device ID
     */
    int get_current_device() const {
        return current_device_;
    }

    /**
     * @brief Check if tensor core is supported on current device
     * @return true if supported, false otherwise
     */
    bool is_tensor_core_supported() const {
        if (!initialized_) return false;
        
        try {
            const auto& info = get_device_info(current_device_);
            return info.tensor_core_support;
        } catch (const std::exception&) {
            return false;
        }
    }

    /**
     * @brief Get compute capability of current device
     * @return Compute capability
     */
    ComputeCapability get_compute_capability() const {
        if (!initialized_) return ComputeCapability::SM_70;
        
        try {
            const auto& info = get_device_info(current_device_);
            return info.compute_capability;
        } catch (const std::exception&) {
            return ComputeCapability::SM_70;
        }
    }

    /**
     * @brief Synchronize current device
     */
    void synchronize() {
        if (initialized_) {
            cudaDeviceSynchronize();
        }
    }

    /**
     * @brief Check if device manager is initialized
     * @return true if initialized, false otherwise
     */
    bool is_initialized() const {
        return initialized_;
    }

private:
    DeviceManager() = default;
    ~DeviceManager() {
        if (initialized_) {
            finalize();
        }
    }
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    bool initialized_ = false;
    int device_count_ = 0;
    int current_device_ = 0;
    std::vector<DeviceInfo> device_infos_;

    /**
     * @brief Query device information
     * @param device_id Device ID
     * @return Device information
     */
    DeviceInfo query_device_info(int device_id) {
        DeviceInfo info;
        info.device_id = device_id;
        
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        
        info.name = std::string(props.name);
        info.multiprocessor_count = props.multiProcessorCount;
        info.max_threads_per_block = props.maxThreadsPerBlock;
        info.max_shared_memory_per_block = props.sharedMemPerBlock;
        info.warp_size = props.warpSize;
        info.total_memory = props.totalGlobalMem;
        
        // Get current free memory
        size_t free_mem, total_mem;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;
        
        // Determine compute capability
        int capability = props.major * 10 + props.minor;
        switch (capability) {
            case 70: info.compute_capability = ComputeCapability::SM_70; break;
            case 75: info.compute_capability = ComputeCapability::SM_75; break;
            case 80: info.compute_capability = ComputeCapability::SM_80; break;
            case 86: info.compute_capability = ComputeCapability::SM_86; break;
            case 89: info.compute_capability = ComputeCapability::SM_89; break;
            case 90: info.compute_capability = ComputeCapability::SM_90; break;
            default: info.compute_capability = ComputeCapability::SM_70; break;
        }
        
        // Check tensor core support (SM 7.0+)
        info.tensor_core_support = (props.major >= 7);
        
        return info;
    }

    /**
     * @brief Check if device meets minimum requirements
     * @param device_id Device ID
     * @return true if device is supported, false otherwise
     */
    bool is_device_supported(int device_id) {
        cudaDeviceProp props;
        cudaError_t error = cudaGetDeviceProperties(&props, device_id);
        if (error != cudaSuccess) return false;
        
        // Check minimum compute capability
        int capability = props.major * 10 + props.minor;
        return capability >= static_cast<int>(config::MIN_COMPUTE_CAPABILITY);
    }
};

// Convenience functions
/**
 * @brief Initialize device subsystem
 * @return true if successful, false otherwise
 */
inline bool initialize() {
    return DeviceManager::instance().initialize();
}

/**
 * @brief Finalize device subsystem
 */
inline void finalize() {
    DeviceManager::instance().finalize();
}

/**
 * @brief Get current device ID
 * @return Current device ID
 */
inline int get_current_device() {
    return DeviceManager::instance().get_current_device();
}

/**
 * @brief Set current device
 * @param device_id Device ID
 * @return true if successful, false otherwise
 */
inline bool set_device(int device_id) {
    return DeviceManager::instance().set_device(device_id);
}

/**
 * @brief Synchronize current device
 */
inline void synchronize() {
    DeviceManager::instance().synchronize();
}

/**
 * @brief Check if tensor core is supported
 * @return true if supported, false otherwise
 */
inline bool is_tensor_core_supported() {
    return DeviceManager::instance().is_tensor_core_supported();
}

/**
 * @brief Get compute capability of current device
 * @return Compute capability
 */
inline ComputeCapability get_compute_capability() {
    return DeviceManager::instance().get_compute_capability();
}

/**
 * @brief Get device count
 * @return Number of supported devices
 */
inline int get_device_count() {
    return DeviceManager::instance().get_device_count();
}

/**
 * @brief Get device information
 * @param device_id Device ID (default: current device)
 * @return Device information
 */
inline const DeviceInfo& get_device_info(int device_id = -1) {
    if (device_id == -1) {
        device_id = get_current_device();
    }
    return DeviceManager::instance().get_device_info(device_id);
}

} // namespace device
} // namespace choreo_ir

#endif // CHOREO_IR_CORE_DEVICE_HPP 