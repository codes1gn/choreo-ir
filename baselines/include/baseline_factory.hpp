/**
 * @file baseline_factory.hpp
 * @brief Factory for creating baseline implementations
 */

#pragma once

#include "baselines.hpp"
#include <memory>
#include <string>
#include <vector>

namespace choreo_ir {
namespace baselines {

/**
 * @brief Factory for creating baseline implementations
 */
class BaselineFactory {
public:
    /**
     * @brief Create a baseline by name
     */
    static std::unique_ptr<Baseline> create(const std::string& name);
    
    /**
     * @brief Create a cuBLAS baseline with specific layout
     */
    static std::unique_ptr<CublasBaseline> create_cublas(DataLayout layout = DataLayout::COLUMN_MAJOR);
    
    /**
     * @brief Create a CUDA baseline
     */
    static std::unique_ptr<CudaBaseline> create_cuda();
    
    /**
     * @brief Create a CUDA baseline with specific layout
     */
    static std::unique_ptr<CudaBaseline> create_cuda(DataLayout layout);
    
    /**
     * @brief Get list of available baselines
     */
    static std::vector<std::string> available_baselines();
};

} // namespace baselines
} // namespace choreo_ir 