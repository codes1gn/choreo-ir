/**
 * @file debug.hpp
 * @brief Debug utilities and assertion macros
 */

#ifndef CHOREO_IR_UTILS_DEBUG_HPP
#define CHOREO_IR_UTILS_DEBUG_HPP

#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include "../core/config.hpp"

namespace choreo_ir {
namespace debug {

/**
 * @enum LogLevel
 * @brief Log message levels
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

/**
 * @brief Get log level name
 * @param level Log level
 * @return Log level name
 */
inline const char* get_log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Print log message
 * @param level Log level
 * @param file Source file name
 * @param line Source line number
 * @param message Log message
 */
inline void print_log(LogLevel level, const char* file, int line, const std::string& message) {
    if (config::DEBUG_MODE) {
        std::cerr << "[" << get_log_level_name(level) << "] " 
                  << file << ":" << line << " - " << message << std::endl;
    }
}

/**
 * @brief Format log message
 * @param args Arguments to format
 * @return Formatted message
 */
template<typename... Args>
std::string format_message(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

// Debug logging macros
#define CHOREO_DEBUG(...)                                                      \
    do {                                                                       \
        if (config::DEBUG_MODE) {                                             \
            choreo_ir::debug::print_log(choreo_ir::debug::LogLevel::DEBUG,    \
                                       __FILE__, __LINE__,                     \
                                       choreo_ir::debug::format_message(__VA_ARGS__)); \
        }                                                                      \
    } while (0)

#define CHOREO_INFO(...)                                                       \
    do {                                                                       \
        if (config::DEBUG_MODE) {                                             \
            choreo_ir::debug::print_log(choreo_ir::debug::LogLevel::INFO,     \
                                       __FILE__, __LINE__,                     \
                                       choreo_ir::debug::format_message(__VA_ARGS__)); \
        }                                                                      \
    } while (0)

#define CHOREO_WARNING(...)                                                    \
    do {                                                                       \
        if (config::DEBUG_MODE) {                                             \
            choreo_ir::debug::print_log(choreo_ir::debug::LogLevel::WARNING,  \
                                       __FILE__, __LINE__,                     \
                                       choreo_ir::debug::format_message(__VA_ARGS__)); \
        }                                                                      \
    } while (0)

#define CHOREO_ERROR(...)                                                      \
    do {                                                                       \
        choreo_ir::debug::print_log(choreo_ir::debug::LogLevel::ERROR,        \
                                   __FILE__, __LINE__,                         \
                                   choreo_ir::debug::format_message(__VA_ARGS__)); \
    } while (0)

// Assertion macros
#define CHOREO_ASSERT(condition, ...)                                         \
    do {                                                                       \
        if (config::ENABLE_ASSERTIONS && !(condition)) {                      \
            CHOREO_ERROR("Assertion failed: " #condition " - ",               \
                        choreo_ir::debug::format_message(__VA_ARGS__));       \
            assert(condition);                                                 \
        }                                                                      \
    } while (0)

#define CHOREO_ASSERT_MSG(condition, message)                                 \
    do {                                                                       \
        if (config::ENABLE_ASSERTIONS && !(condition)) {                      \
            CHOREO_ERROR("Assertion failed: " #condition " - " message);      \
            assert(condition);                                                 \
        }                                                                      \
    } while (0)

// Device assertions (for use in CUDA kernels)
#define CHOREO_DEVICE_ASSERT(condition)                                       \
    do {                                                                       \
        if (config::ENABLE_ASSERTIONS) {                                      \
            assert(condition);                                                 \
        }                                                                      \
    } while (0)

/**
 * @brief Check tensor dimensions match
 * @param tensor1_shape First tensor shape
 * @param tensor2_shape Second tensor shape
 * @param operation Operation name
 */
template<typename Shape1, typename Shape2>
void check_tensor_dimensions(const Shape1& tensor1_shape, const Shape2& tensor2_shape, 
                           const std::string& operation) {
    CHOREO_ASSERT(tensor1_shape.ndims() == tensor2_shape.ndims(),
                 "Tensor dimension mismatch in ", operation, ": ",
                 tensor1_shape.ndims(), " vs ", tensor2_shape.ndims());
    
    for (dim_t i = 0; i < tensor1_shape.ndims(); ++i) {
        CHOREO_ASSERT(tensor1_shape[i] == tensor2_shape[i],
                     "Tensor shape mismatch in ", operation, " at dimension ", i, ": ",
                     tensor1_shape[i], " vs ", tensor2_shape[i]);
    }
}

/**
 * @brief Check matrix multiplication dimensions
 * @param a_shape Shape of matrix A
 * @param b_shape Shape of matrix B
 */
template<typename Shape>
void check_matmul_dimensions(const Shape& a_shape, const Shape& b_shape) {
    CHOREO_ASSERT(a_shape.ndims() >= 2, "Matrix A must have at least 2 dimensions");
    CHOREO_ASSERT(b_shape.ndims() >= 2, "Matrix B must have at least 2 dimensions");
    
    index_t a_cols = a_shape.last_dim();
    index_t b_rows = b_shape.second_last_dim();
    
    CHOREO_ASSERT(a_cols == b_rows,
                 "Matrix multiplication dimension mismatch: A columns (", a_cols, 
                 ") != B rows (", b_rows, ")");
}

/**
 * @brief Check tensor is not empty
 * @param tensor_shape Tensor shape
 * @param operation Operation name
 */
template<typename Shape>
void check_tensor_not_empty(const Shape& tensor_shape, const std::string& operation) {
    CHOREO_ASSERT(!tensor_shape.empty(),
                 "Tensor is empty in operation: ", operation);
}

/**
 * @brief Check tensor is contiguous
 * @param tensor_layout Tensor layout
 * @param operation Operation name
 */
template<typename Layout>
void check_tensor_contiguous(const Layout& tensor_layout, const std::string& operation) {
    CHOREO_ASSERT(tensor_layout.is_contiguous(),
                 "Tensor must be contiguous for operation: ", operation);
}

/**
 * @brief Check tensor core compatibility
 * @param tensor_layout Tensor layout
 * @param operation Operation name
 */
template<typename Layout>
void check_tensor_core_compatibility(const Layout& tensor_layout, const std::string& operation) {
    CHOREO_ASSERT(tensor_layout.is_tensor_core_compatible(),
                 "Tensor layout is not compatible with tensor core for operation: ", operation);
}

/**
 * @brief Check data type compatibility
 * @param dtype1 First data type
 * @param dtype2 Second data type
 * @param operation Operation name
 */
inline void check_dtype_compatibility(DataType dtype1, DataType dtype2, const std::string& operation) {
    CHOREO_ASSERT(dtype1 == dtype2,
                 "Data type mismatch in ", operation, ": ",
                 static_cast<int>(dtype1), " vs ", static_cast<int>(dtype2));
}

/**
 * @brief Performance warning for non-optimal layouts
 * @param layout Tensor layout
 * @param operation Operation name
 */
template<typename Layout>
void warn_non_optimal_layout(const Layout& layout, const std::string& operation) {
    if (!layout.is_coalesced()) {
        CHOREO_WARNING("Non-coalesced memory access in ", operation, 
                      " may result in poor performance");
    }
}

/**
 * @brief Benchmark timer for performance debugging
 */
class BenchmarkTimer {
public:
    BenchmarkTimer(const std::string& name) : name_(name) {
        if (config::DEBUG_MODE) {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    ~BenchmarkTimer() {
        if (config::DEBUG_MODE) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count();
            CHOREO_DEBUG("Timer [", name_, "]: ", duration, " microseconds");
        }
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Benchmark timer macro
#define CHOREO_BENCHMARK(name) \
    choreo_ir::debug::BenchmarkTimer timer(name)

} // namespace debug
} // namespace choreo_ir

#endif // CHOREO_IR_UTILS_DEBUG_HPP 