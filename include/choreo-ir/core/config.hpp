/**
 * @file config.hpp
 * @brief Configuration constants and compile-time options for Choreo-IR
 */

#ifndef CHOREO_IR_CORE_CONFIG_HPP
#define CHOREO_IR_CORE_CONFIG_HPP

#include "types.hpp"

namespace choreo_ir {
namespace config {

// Version information
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Default tensor configuration
constexpr dim_t MAX_TENSOR_DIMS = 8;
constexpr size_t DEFAULT_ALIGNMENT = 256;  // 256-byte alignment for optimal memory access

// CUDA configuration
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;

// Shared memory configuration
constexpr size_t MAX_SHARED_MEMORY_PER_BLOCK = 48 * 1024;  // 48KB for modern GPUs
constexpr size_t SHARED_MEMORY_BANK_SIZE = 32;  // 32-bit banks

// Tensor Core configuration
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Performance thresholds
constexpr float PERFORMANCE_TOLERANCE = 0.05f;  // 5% performance tolerance

// Debug configuration
#ifdef CHOREO_IR_DEBUG
constexpr bool DEBUG_MODE = true;
#else
constexpr bool DEBUG_MODE = false;
#endif

// Profiling configuration
#ifdef CHOREO_IR_PROFILE
constexpr bool PROFILE_MODE = true;
#else
constexpr bool PROFILE_MODE = false;
#endif

// Default data type for operations
constexpr DataType DEFAULT_DTYPE = DataType::FLOAT16;

// Memory layout preferences
enum class LayoutPreference {
    ROW_MAJOR,
    COLUMN_MAJOR,
    AUTO  // Let the library choose optimal layout
};

constexpr LayoutPreference DEFAULT_LAYOUT = LayoutPreference::AUTO;

// Compute capability detection
constexpr ComputeCapability MIN_COMPUTE_CAPABILITY = ComputeCapability::SM_70;

// Error handling configuration
constexpr bool ENABLE_CUDA_ERROR_CHECKING = true;
constexpr bool ENABLE_ASSERTIONS = DEBUG_MODE;

// Optimization flags
constexpr bool ENABLE_FAST_MATH = true;
constexpr bool ENABLE_TENSOR_CORE = true;
constexpr bool ENABLE_ASYNC_EXECUTION = true;

} // namespace config
} // namespace choreo_ir

#endif // CHOREO_IR_CORE_CONFIG_HPP 