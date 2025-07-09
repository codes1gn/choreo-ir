/**
 * @file choreo-ir.hpp
 * @brief Main header file for the Choreo-IR CUDA tensor abstraction library
 * 
 * This is a zero-cost abstraction library for CUDA tensor operations,
 * providing high-level programming interfaces while maintaining optimal performance.
 * 
 * @version 1.0.0
 * @author Choreo-IR Team
 */

#ifndef CHOREO_IR_HPP
#define CHOREO_IR_HPP

// Core components
#include "core/types.hpp"
#include "core/config.hpp"
#include "core/device.hpp"

// Tensor abstraction
#include "tensor/tensor.hpp"
#include "tensor/shape.hpp"
#include "tensor/stride.hpp"
#include "tensor/layout.hpp"

// Compute operations
#include "compute/matmul.hpp"
#include "compute/conv.hpp"
#include "compute/elementwise.hpp"

// Utilities
#include "utils/debug.hpp"
#include "utils/profiler.hpp"
#include "utils/cuda_utils.hpp"

/**
 * @namespace choreo_ir
 * @brief Main namespace for the Choreo-IR library
 */
namespace choreo_ir {

/**
 * @brief Initialize the Choreo-IR library
 * @return true if initialization successful, false otherwise
 */
inline bool initialize() {
    return device::initialize();
}

/**
 * @brief Finalize the Choreo-IR library
 */
inline void finalize() {
    device::finalize();
}

} // namespace choreo_ir

#endif // CHOREO_IR_HPP 