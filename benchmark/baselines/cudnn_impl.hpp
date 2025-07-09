/**
 * @file cudnn_impl.hpp
 * @brief cuDNN baseline implementation stub
 */

#pragma once

#include <cudnn.h>

namespace choreo_ir {
namespace baselines {

/**
 * @brief cuDNN baseline implementation placeholder
 */
class CudnnBaseline {
public:
    CudnnBaseline() = default;
    ~CudnnBaseline() = default;
    
    // Placeholder methods
    bool initialize() { return true; }
    void finalize() {}
};

} // namespace baselines
} // namespace choreo_ir 