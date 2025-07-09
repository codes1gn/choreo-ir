/**
 * @file conv.hpp
 * @brief Convolution operations interface definitions
 */

#ifndef CHOREO_IR_COMPUTE_CONV_HPP
#define CHOREO_IR_COMPUTE_CONV_HPP

#include <cuda_runtime.h>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @enum ConvAlgorithm
 * @brief Convolution algorithm types
 */
enum class ConvAlgorithm {
    AUTO,           // Automatically select best algorithm
    DIRECT,         // Direct convolution
    IMPLICIT_GEMM,  // Implicit GEMM convolution
    WINOGRAD,       // Winograd convolution
    FFT,            // FFT-based convolution
    CUDNN           // cuDNN library
};

/**
 * @struct ConvConfig
 * @brief Configuration for convolution operations
 */
struct ConvConfig {
    ConvAlgorithm algorithm = ConvAlgorithm::AUTO;
    dim_t stride_h = 1;
    dim_t stride_w = 1;
    dim_t pad_h = 0;
    dim_t pad_w = 0;
    dim_t dilation_h = 1;
    dim_t dilation_w = 1;
    dim_t groups = 1;
    bool use_tensor_core = true;
    
    ConvConfig() = default;
};

/**
 * @brief 2D Convolution interface: output = conv2d(input, weight, bias)
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (K, C, R, S)
 * @param bias Bias tensor (K,) - optional
 * @param output Output tensor (N, K, H_out, W_out)
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void conv2d(const TensorView<T>& input, const TensorView<T>& weight,
           const TensorView<T>& bias, TensorView<T>& output,
           const ConvConfig& config = ConvConfig(), cudaStream_t stream = 0);

/**
 * @brief 2D Convolution without bias
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (K, C, R, S)
 * @param output Output tensor (N, K, H_out, W_out)
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void conv2d(const TensorView<T>& input, const TensorView<T>& weight,
           TensorView<T>& output, const ConvConfig& config = ConvConfig(),
           cudaStream_t stream = 0);

/**
 * @brief 2D Convolution with automatic output allocation
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (K, C, R, S)
 * @param bias Bias tensor (K,) - optional
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @return Output tensor
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> conv2d(const TensorView<T>& input, const TensorView<T>& weight,
                const TensorView<T>& bias, const ConvConfig& config = ConvConfig(),
                cudaStream_t stream = 0);

/**
 * @brief 2D Convolution without bias and with automatic output allocation
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (K, C, R, S)
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @return Output tensor
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> conv2d(const TensorView<T>& input, const TensorView<T>& weight,
                const ConvConfig& config = ConvConfig(), cudaStream_t stream = 0);

/**
 * @brief Depthwise convolution
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (C, 1, R, S)
 * @param output Output tensor (N, C, H_out, W_out)
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void depthwise_conv2d(const TensorView<T>& input, const TensorView<T>& weight,
                     TensorView<T>& output, const ConvConfig& config = ConvConfig(),
                     cudaStream_t stream = 0);

/**
 * @brief Pointwise convolution (1x1 convolution)
 * @param input Input tensor (N, C, H, W)
 * @param weight Weight tensor (K, C, 1, 1)
 * @param output Output tensor (N, K, H, W)
 * @param config Convolution configuration
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void pointwise_conv2d(const TensorView<T>& input, const TensorView<T>& weight,
                     TensorView<T>& output, const ConvConfig& config = ConvConfig(),
                     cudaStream_t stream = 0);

/**
 * @brief Calculate output dimensions for convolution
 * @param input_h Input height
 * @param input_w Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param dilation_h Dilation height
 * @param dilation_w Dilation width
 * @return Output dimensions (height, width)
 */
inline std::pair<index_t, index_t> calculate_conv_output_size(
    index_t input_h, index_t input_w,
    index_t kernel_h, index_t kernel_w,
    index_t stride_h, index_t stride_w,
    index_t pad_h, index_t pad_w,
    index_t dilation_h = 1, index_t dilation_w = 1) {
    
    index_t output_h = (input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    index_t output_w = (input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    return {output_h, output_w};
}

/**
 * @brief Check if tensor core can be used for convolution
 * @param input_shape Input tensor shape
 * @param weight_shape Weight tensor shape
 * @param config Convolution configuration
 * @return true if tensor core can be used, false otherwise
 */
template<typename T>
bool can_use_tensor_core_conv(const Shape& input_shape, const Shape& weight_shape,
                              const ConvConfig& config);

/**
 * @brief Get optimal configuration for convolution
 * @param input_shape Input tensor shape
 * @param weight_shape Weight tensor shape
 * @param config Base configuration
 * @return Optimal configuration
 */
template<typename T>
ConvConfig get_optimal_conv_config(const Shape& input_shape, const Shape& weight_shape,
                                  const ConvConfig& config = ConvConfig());

/**
 * @brief Estimate TFLOPS for convolution
 * @param batch_size Batch size
 * @param out_channels Output channels
 * @param out_height Output height
 * @param out_width Output width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param in_channels Input channels
 * @param time_ms Execution time in milliseconds
 * @return TFLOPS
 */
inline float estimate_conv_tflops(index_t batch_size, index_t out_channels,
                                 index_t out_height, index_t out_width,
                                 index_t kernel_h, index_t kernel_w,
                                 index_t in_channels, float time_ms) {
    if (time_ms <= 0.0f) return 0.0f;
    
    size_t flops = 2ULL * batch_size * out_channels * out_height * out_width *
                   kernel_h * kernel_w * in_channels;
    return (flops / 1e12f) / (time_ms / 1000.0f);
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_CONV_HPP 