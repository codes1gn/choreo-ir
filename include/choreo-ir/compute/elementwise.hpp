/**
 * @file elementwise.hpp
 * @brief Element-wise operations interface definitions
 */

#ifndef CHOREO_IR_COMPUTE_ELEMENTWISE_HPP
#define CHOREO_IR_COMPUTE_ELEMENTWISE_HPP

#include <cuda_runtime.h>
#include <functional>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @enum ElementwiseOp
 * @brief Element-wise operation types
 */
enum class ElementwiseOp {
    ADD,        // Addition
    SUB,        // Subtraction
    MUL,        // Multiplication
    DIV,        // Division
    POW,        // Power
    MAX,        // Maximum
    MIN,        // Minimum
    SQRT,       // Square root
    EXP,        // Exponential
    LOG,        // Natural logarithm
    SIN,        // Sine
    COS,        // Cosine
    TANH,       // Hyperbolic tangent
    RELU,       // ReLU activation
    GELU,       // GELU activation
    SIGMOID     // Sigmoid activation
};

/**
 * @brief Element-wise binary operation interface: C = A op B
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param op Operation type
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void elementwise_binary(const TensorView<T>& A, const TensorView<T>& B,
                       TensorView<T>& C, ElementwiseOp op, cudaStream_t stream = 0);

/**
 * @brief Element-wise unary operation interface: B = op(A)
 * @param A Input tensor A
 * @param B Output tensor B
 * @param op Operation type
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void elementwise_unary(const TensorView<T>& A, TensorView<T>& B,
                      ElementwiseOp op, cudaStream_t stream = 0);

/**
 * @brief Element-wise scalar operation interface: B = A op scalar
 * @param A Input tensor A
 * @param scalar Scalar value
 * @param B Output tensor B
 * @param op Operation type
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void elementwise_scalar(const TensorView<T>& A, T scalar, TensorView<T>& B,
                       ElementwiseOp op, cudaStream_t stream = 0);

/**
 * @brief Element-wise addition interface: C = A + B
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void add(const TensorView<T>& A, const TensorView<T>& B, TensorView<T>& C,
         cudaStream_t stream = 0);

/**
 * @brief Element-wise addition with automatic output allocation
 * @param A Input tensor A
 * @param B Input tensor B
 * @param stream CUDA stream
 * @return Output tensor C
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> add(const TensorView<T>& A, const TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief Element-wise subtraction interface: C = A - B
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void sub(const TensorView<T>& A, const TensorView<T>& B, TensorView<T>& C,
         cudaStream_t stream = 0);

/**
 * @brief Element-wise subtraction with automatic output allocation
 * @param A Input tensor A
 * @param B Input tensor B
 * @param stream CUDA stream
 * @return Output tensor C
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> sub(const TensorView<T>& A, const TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief Element-wise multiplication interface: C = A * B
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void mul(const TensorView<T>& A, const TensorView<T>& B, TensorView<T>& C,
         cudaStream_t stream = 0);

/**
 * @brief Element-wise multiplication with automatic output allocation
 * @param A Input tensor A
 * @param B Input tensor B
 * @param stream CUDA stream
 * @return Output tensor C
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> mul(const TensorView<T>& A, const TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief Element-wise division interface: C = A / B
 * @param A Input tensor A
 * @param B Input tensor B
 * @param C Output tensor C
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void div(const TensorView<T>& A, const TensorView<T>& B, TensorView<T>& C,
         cudaStream_t stream = 0);

/**
 * @brief Element-wise division with automatic output allocation
 * @param A Input tensor A
 * @param B Input tensor B
 * @param stream CUDA stream
 * @return Output tensor C
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> div(const TensorView<T>& A, const TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief ReLU activation interface: B = max(0, A)
 * @param A Input tensor A
 * @param B Output tensor B
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void relu(const TensorView<T>& A, TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief ReLU activation with automatic output allocation
 * @param A Input tensor A
 * @param stream CUDA stream
 * @return Output tensor B
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> relu(const TensorView<T>& A, cudaStream_t stream = 0);

/**
 * @brief GELU activation interface: B = 0.5 * A * (1 + tanh(sqrt(2/π) * (A + 0.044715 * A^3)))
 * @param A Input tensor A
 * @param B Output tensor B
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void gelu(const TensorView<T>& A, TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief GELU activation with automatic output allocation
 * @param A Input tensor A
 * @param stream CUDA stream
 * @return Output tensor B
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> gelu(const TensorView<T>& A, cudaStream_t stream = 0);

/**
 * @brief Sigmoid activation interface: B = 1 / (1 + exp(-A))
 * @param A Input tensor A
 * @param B Output tensor B
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void sigmoid(const TensorView<T>& A, TensorView<T>& B, cudaStream_t stream = 0);

/**
 * @brief Sigmoid activation with automatic output allocation
 * @param A Input tensor A
 * @param stream CUDA stream
 * @return Output tensor B
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
Tensor<T> sigmoid(const TensorView<T>& A, cudaStream_t stream = 0);

/**
 * @brief Attention scores computation interface: scores = Q * K^T * scale
 * @param Q Query tensor
 * @param K Key tensor
 * @param scale Scaling factor
 * @param scores Output scores tensor
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void attention_scores(const TensorView<T>& Q, const TensorView<T>& K,
                     T scale, TensorView<T>& scores, cudaStream_t stream = 0);

/**
 * @brief Fused scale and softmax operation interface
 * @param input Input tensor
 * @param scale Scaling factor
 * @param output Output tensor
 * @param stream CUDA stream
 * @note This is an interface declaration. Implementations are provided in benchmark/
 */
template<typename T>
void fused_scale_softmax(const TensorView<T>& input, T scale,
                        TensorView<T>& output, cudaStream_t stream = 0);

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_ELEMENTWISE_HPP 