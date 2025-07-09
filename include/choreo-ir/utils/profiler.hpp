/**
 * @file profiler.hpp
 * @brief Performance profiler with CUDA events and Nsight integration
 */

#ifndef CHOREO_IR_UTILS_PROFILER_HPP
#define CHOREO_IR_UTILS_PROFILER_HPP

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include "cuda_utils.hpp"
#include "../core/config.hpp"

namespace choreo_ir {
namespace profiler {

/**
 * @struct ProfileResult
 * @brief Result of a profiling measurement
 */
struct ProfileResult {
    std::string name;
    float gpu_time_ms;
    float cpu_time_ms;
    size_t memory_used;
    size_t flops;
    float tflops;
    
    ProfileResult(const std::string& n = "") 
        : name(n), gpu_time_ms(0.0f), cpu_time_ms(0.0f), memory_used(0), flops(0), tflops(0.0f) {}
};

/**
 * @class GPUTimer
 * @brief CUDA event-based GPU timer
 */
class GPUTimer {
public:
    GPUTimer() : start_event_(nullptr), end_event_(nullptr) {
        if (config::PROFILE_MODE) {
            start_event_ = cuda_utils::create_event();
            end_event_ = cuda_utils::create_event();
        }
    }
    
    ~GPUTimer() {
        if (start_event_) cuda_utils::destroy_event(start_event_);
        if (end_event_) cuda_utils::destroy_event(end_event_);
    }
    
    void start(cudaStream_t stream = 0) {
        if (config::PROFILE_MODE) {
            cuda_utils::record_event(start_event_, stream);
        }
    }
    
    void stop(cudaStream_t stream = 0) {
        if (config::PROFILE_MODE) {
            cuda_utils::record_event(end_event_, stream);
        }
    }
    
    float elapsed_time() {
        if (config::PROFILE_MODE) {
            cuda_utils::wait_event(end_event_);
            return cuda_utils::get_elapsed_time(start_event_, end_event_);
        }
        return 0.0f;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
};

/**
 * @class CPUTimer
 * @brief High-resolution CPU timer
 */
class CPUTimer {
public:
    CPUTimer() : start_time_(), end_time_() {}
    
    void start() {
        if (config::PROFILE_MODE) {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    void stop() {
        if (config::PROFILE_MODE) {
            end_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    float elapsed_time() {
        if (config::PROFILE_MODE) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time_ - start_time_);
            return duration.count() / 1000.0f;  // Convert to milliseconds
        }
        return 0.0f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

/**
 * @class NsightProfiler
 * @brief Nsight profiler integration
 */
class NsightProfiler {
public:
    /**
     * @brief Start profiling range
     * @param name Range name
     * @param color Color ID
     */
    static void start_range(const std::string& name, uint32_t color = 0xFF00FF00) {
        if (config::PROFILE_MODE) {
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.colorType = NVTX_COLOR_ARGB;
            eventAttrib.color = color;
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = name.c_str();
            nvtxRangePushEx(&eventAttrib);
        }
    }
    
    /**
     * @brief End profiling range
     */
    static void end_range() {
        if (config::PROFILE_MODE) {
            nvtxRangePop();
        }
    }
    
    /**
     * @brief Mark a point in time
     * @param name Mark name
     * @param color Color ID
     */
    static void mark(const std::string& name, uint32_t color = 0xFFFF0000) {
        if (config::PROFILE_MODE) {
            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.colorType = NVTX_COLOR_ARGB;
            eventAttrib.color = color;
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = name.c_str();
            nvtxMarkEx(&eventAttrib);
        }
    }
};

/**
 * @class ProfileScope
 * @brief RAII profiling scope
 */
class ProfileScope {
public:
    ProfileScope(const std::string& name, cudaStream_t stream = 0) 
        : name_(name), stream_(stream) {
        if (config::PROFILE_MODE) {
            cpu_timer_.start();
            gpu_timer_.start(stream);
            NsightProfiler::start_range(name);
        }
    }
    
    ~ProfileScope() {
        if (config::PROFILE_MODE) {
            gpu_timer_.stop(stream_);
            cpu_timer_.stop();
            NsightProfiler::end_range();
            
            ProfileResult result(name_);
            result.gpu_time_ms = gpu_timer_.elapsed_time();
            result.cpu_time_ms = cpu_timer_.elapsed_time();
            
            Profiler::instance().add_result(result);
        }
    }

private:
    std::string name_;
    cudaStream_t stream_;
    GPUTimer gpu_timer_;
    CPUTimer cpu_timer_;
};

/**
 * @class Profiler
 * @brief Main profiler class
 */
class Profiler {
public:
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }
    
    void add_result(const ProfileResult& result) {
        if (config::PROFILE_MODE) {
            results_[result.name].push_back(result);
        }
    }
    
    void clear_results() {
        results_.clear();
    }
    
    std::vector<ProfileResult> get_results(const std::string& name) const {
        auto it = results_.find(name);
        if (it != results_.end()) {
            return it->second;
        }
        return {};
    }
    
    void print_summary() const {
        if (!config::PROFILE_MODE) return;
        
        std::cout << "\n=== Profiling Summary ===" << std::endl;
        std::cout << std::left << std::setw(30) << "Operation" 
                  << std::setw(15) << "GPU Time (ms)" 
                  << std::setw(15) << "CPU Time (ms)" 
                  << std::setw(15) << "TFLOPS" << std::endl;
        std::cout << std::string(75, '-') << std::endl;
        
        for (const auto& pair : results_) {
            const auto& name = pair.first;
            const auto& results = pair.second;
            
            if (results.empty()) continue;
            
            float avg_gpu_time = 0.0f;
            float avg_cpu_time = 0.0f;
            float avg_tflops = 0.0f;
            
            for (const auto& result : results) {
                avg_gpu_time += result.gpu_time_ms;
                avg_cpu_time += result.cpu_time_ms;
                avg_tflops += result.tflops;
            }
            
            avg_gpu_time /= results.size();
            avg_cpu_time /= results.size();
            avg_tflops /= results.size();
            
            std::cout << std::left << std::setw(30) << name
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg_gpu_time
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg_cpu_time
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg_tflops
                      << std::endl;
        }
        std::cout << std::string(75, '-') << std::endl;
    }
    
    void save_to_file(const std::string& filename) const {
        if (!config::PROFILE_MODE) return;
        
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "operation,gpu_time_ms,cpu_time_ms,tflops\n";
        for (const auto& pair : results_) {
            const auto& name = pair.first;
            const auto& results = pair.second;
            
            for (const auto& result : results) {
                file << name << "," << result.gpu_time_ms << "," 
                     << result.cpu_time_ms << "," << result.tflops << "\n";
            }
        }
    }

private:
    Profiler() = default;
    ~Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    
    std::unordered_map<std::string, std::vector<ProfileResult>> results_;
};

/**
 * @brief Calculate TFLOPS for matrix multiplication
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param time_ms Time in milliseconds
 * @return TFLOPS
 */
inline float calculate_matmul_tflops(index_t m, index_t n, index_t k, float time_ms) {
    if (time_ms <= 0.0f) return 0.0f;
    
    // Matrix multiplication: C = A * B requires 2*M*N*K operations
    size_t flops = 2ULL * m * n * k;
    float tflops = (flops / 1e12f) / (time_ms / 1000.0f);
    return tflops;
}

/**
 * @brief Calculate TFLOPS for convolution
 * @param batch_size Batch size
 * @param out_channels Output channels
 * @param out_height Output height
 * @param out_width Output width
 * @param kernel_size Kernel size
 * @param in_channels Input channels
 * @param time_ms Time in milliseconds
 * @return TFLOPS
 */
inline float calculate_conv_tflops(index_t batch_size, index_t out_channels, 
                                  index_t out_height, index_t out_width,
                                  index_t kernel_size, index_t in_channels, 
                                  float time_ms) {
    if (time_ms <= 0.0f) return 0.0f;
    
    // Convolution: each output element requires kernel_size^2 * in_channels operations
    size_t flops = 2ULL * batch_size * out_channels * out_height * out_width * 
                   kernel_size * kernel_size * in_channels;
    float tflops = (flops / 1e12f) / (time_ms / 1000.0f);
    return tflops;
}

// Profiling macros
#define CHOREO_PROFILE(name) \
    choreo_ir::profiler::ProfileScope profile_scope(name)

#define CHOREO_PROFILE_STREAM(name, stream) \
    choreo_ir::profiler::ProfileScope profile_scope(name, stream)

#define CHOREO_NVTX_RANGE(name) \
    choreo_ir::profiler::NsightProfiler::start_range(name); \
    auto nvtx_guard = std::unique_ptr<void, void(*)(void*)>(nullptr, [](void*){ \
        choreo_ir::profiler::NsightProfiler::end_range(); \
    })

#define CHOREO_NVTX_MARK(name) \
    choreo_ir::profiler::NsightProfiler::mark(name)

} // namespace profiler
} // namespace choreo_ir

#endif // CHOREO_IR_UTILS_PROFILER_HPP 