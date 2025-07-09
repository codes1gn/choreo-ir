/**
 * @file matmul.hpp
 * @brief Tensor core operations and matrix computation abstractions
 */

#ifndef CHOREO_IR_COMPUTE_MATMUL_HPP
#define CHOREO_IR_COMPUTE_MATMUL_HPP

#include <cuda_runtime.h>
#include <mma.h>
#include "../tensor/tensor.hpp"
#include "../core/types.hpp"
#include "../core/config.hpp"

namespace choreo_ir {
namespace compute {

/**
 * @enum TensorCoreType
 * @brief Supported tensor core instruction types
 */
enum class TensorCoreType {
    WMMA_16x16x16,  // WMMA 16x16x16
    MMA_16x8x16,    // MMA 16x8x16 (Ampere+)
    MMA_16x8x32,    // MMA 16x8x32 (Ampere+)
    AUTO            // Automatically select based on hardware
};

/**
 * @brief WMMA fragment wrapper for type safety
 */
template<typename T, int M, int N, int K>
class WmmaFragment {
public:
    using fragment_type = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, T, nvcuda::wmma::row_major>;
    
    __device__ WmmaFragment() = default;
    
    __device__ void load_matrix(const T* ptr, index_t ldm) {
        nvcuda::wmma::load_matrix_sync(frag_, ptr, ldm);
    }
    
    __device__ void store_matrix(T* ptr, index_t ldm) const {
        nvcuda::wmma::store_matrix_sync(ptr, frag_, ldm, nvcuda::wmma::layout_t::row_major);
    }
    
    __device__ fragment_type& get() { return frag_; }
    __device__ const fragment_type& get() const { return frag_; }
    
private:
    fragment_type frag_;
};

/**
 * @brief Tensor core matrix-matrix accumulate operation
 * @param A Matrix A fragment
 * @param B Matrix B fragment  
 * @param C Accumulator fragment (input/output)
 */
template<typename T, int M, int N, int K>
__device__ void mma_sync(const WmmaFragment<T, M, N, K>& A,
                        const WmmaFragment<T, M, N, K>& B,
                        WmmaFragment<float, M, N, K>& C) {
    nvcuda::wmma::mma_sync(C.get(), A.get(), B.get(), C.get());
}

/**
 * @brief Fill fragment with a constant value
 * @param frag Fragment to fill
 * @param value Fill value
 */
template<typename T, int M, int N, int K>
__device__ void fill_fragment(WmmaFragment<T, M, N, K>& frag, T value) {
    nvcuda::wmma::fill_fragment(frag.get(), value);
}

/**
 * @brief Check if current thread can use tensor cores
 * @return true if tensor cores are available
 */
__device__ inline bool can_use_tensor_cores() {
    return __CUDA_ARCH__ >= 700; // Volta and newer
}

/**
 * @brief Get optimal tensor core type for current architecture
 * @return Recommended tensor core type
 */
__device__ inline TensorCoreType get_optimal_tensor_core_type() {
    if (__CUDA_ARCH__ >= 800) {
        return TensorCoreType::MMA_16x8x16; // Ampere+
    } else if (__CUDA_ARCH__ >= 700) {
        return TensorCoreType::WMMA_16x16x16; // Volta+
    }
    return TensorCoreType::AUTO;
}

/**
 * @brief Synchronize warp threads for tensor core operations
 */
__device__ inline void warp_sync() {
    __syncwarp();
}

/**
 * @brief Matrix multiplication using tensor cores (16x16x16 WMMA)
 * @param A Input tensor A (must be in shared/register memory)
 * @param B Input tensor B (must be in shared/register memory)
 * @param C Output tensor C (accumulator)
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for existing C values
 */
template<typename T>
__device__ void tensor_core_mma_16x16x16(const TensorView<T>& A, 
                                        const TensorView<T>& B,
                                        TensorView<float>& C,
                                        float alpha = 1.0f, 
                                        float beta = 0.0f) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, 16, 16, 16, T, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, T, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // Load fragments
    load_matrix_sync(a_frag, A.data(), A.stride()[0]);
    load_matrix_sync(b_frag, B.data(), B.stride()[0]);
    
    if (beta != 0.0f) {
        load_matrix_sync(c_frag, C.data(), C.stride()[0]);
    } else {
        fill_fragment(c_frag, 0.0f);
    }
    
    // Perform MMA
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Scale if needed
    if (alpha != 1.0f || beta != 1.0f) {
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * c_frag.x[i];
        }
    }
    
    // Store result
    store_matrix_sync(C.data(), c_frag, C.stride()[0], layout_t::row_major);
}

/**
 * @brief Cooperative matrix multiply-accumulate using thread block
 * @param A Input tensor A  
 * @param B Input tensor B
 * @param C Output tensor C
 * @param tile_m Tile size in M dimension
 * @param tile_n Tile size in N dimension  
 * @param tile_k Tile size in K dimension
 */
template<typename T>
__device__ void cooperative_mma(const TensorView<T>& A,
                               const TensorView<T>& B, 
                               TensorView<float>& C,
                               dim_t tile_m = 16,
                               dim_t tile_n = 16,
                               dim_t tile_k = 16) {
    // This would be implemented using cooperative groups
    // and multiple tensor core instructions
    
    // Calculate thread's tile position
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Each warp handles one 16x16 tile
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) / (tile_m / 16);
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / (tile_n / 16);
    
    // Create local tile views
    auto A_tile = A.tile(Shape({tile_m, tile_k}));
    auto B_tile = B.tile(Shape({tile_k, tile_n}));
    auto C_tile = C.tile(Shape({tile_m, tile_n}));
    
    // Use tensor cores for the tile
    if (can_use_tensor_cores()) {
        tensor_core_mma_16x16x16(A_tile, B_tile, C_tile);
    }
}

} // namespace compute
} // namespace choreo_ir

#endif // CHOREO_IR_COMPUTE_MATMUL_HPP 