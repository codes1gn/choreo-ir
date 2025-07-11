/**
 * @file baseline_factory.cpp
 * @brief Implementation of baseline factory
 */

#include "baseline_factory.hpp"
#include "cublas_baseline.hpp"
#include "cuda_baseline.hpp"
#include <stdexcept>

namespace choreo_ir {
namespace baselines {

std::unique_ptr<Baseline> BaselineFactory::create(const std::string& name) {
    if (name == "cublas" || name == "cuBLAS") {
        return std::make_unique<CublasBaseline>(DataLayout::COLUMN_MAJOR);
    } else if (name == "cublas-row" || name == "cuBLAS-RowMajor") {
        return std::make_unique<CublasBaseline>(DataLayout::ROW_MAJOR);
    } else if (name == "cublas-col" || name == "cuBLAS-ColMajor") {
        return std::make_unique<CublasBaseline>(DataLayout::COLUMN_MAJOR);
    } else if (name == "cuda" || name == "CUDA") {
        return std::make_unique<CudaBaseline>();
    } else {
        throw std::invalid_argument("Unknown baseline: " + name);
    }
}

std::unique_ptr<CublasBaseline> BaselineFactory::create_cublas(DataLayout layout) {
    return std::make_unique<CublasBaseline>(layout);
}

std::unique_ptr<CudaBaseline> BaselineFactory::create_cuda(DataLayout layout) {
    return std::make_unique<CudaBaseline>(layout);
}

std::vector<std::string> BaselineFactory::available_baselines() {
    return {"cublas", "cublas-row", "cublas-col", "cuda"};
}

} // namespace baselines
} // namespace choreo_ir 