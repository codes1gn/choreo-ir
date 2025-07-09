/**
 * @file gemm_example.cu
 * @brief Example of writing CUDA kernels using Choreo-IR tensor abstractions
 * 
 * This demonstrates the programming model where users write kernels using
 * tensor operations like: dst = src.tile(shape) for data movement and
 * tensor core instructions for computation.
 */

#include <cuda_runtime.h>
#include "choreo-ir/choreo-ir.hpp"

using namespace choreo_ir;
using namespace choreo_ir::compute;

/**
 * @brief Matrix multiplication kernel using choreo-ir abstractions
 * 
 * This kernel demonstrates the programming pattern:
 * 1. Define shared and local memory tensors
 * 2. Copy tiles from global to shared: shared_A = global_A.tile(tile_shape)
 * 3. Copy from shared to local/registers for tensor cores
 * 4. Call tensor core instructions
 * 5. Write results back to global memory
 */
template<typename T, dim_t TILE_M, dim_t TILE_N, dim_t TILE_K>
__global__ void gemm_tensor_core_kernel(
    TensorView<T> A,        // Global memory input A
    TensorView<T> B,        // Global memory input B  
    TensorView<float> C,    // Global memory output C
    index_t M, index_t N, index_t K,
    float alpha = 1.0f, float beta = 0.0f) {
    
    // Block and thread indices
    const dim_t block_row = blockIdx.y;
    const dim_t block_col = blockIdx.x;
    const dim_t thread_row = threadIdx.y;
    const dim_t thread_col = threadIdx.x;
    
    // Shared memory tensors for tiles
    __shared__ T shared_A_data[TILE_M * TILE_K];
    __shared__ T shared_B_data[TILE_K * TILE_N];
    
    // Create shared memory tensor views
    auto shared_A = TensorView<T>(shared_A_data, 
                                 Layout(Shape({TILE_M, TILE_K})), 
                                 MemorySpace::SHARED);
    auto shared_B = TensorView<T>(shared_B_data, 
                                 Layout(Shape({TILE_K, TILE_N})), 
                                 MemorySpace::SHARED);
    
    // Local accumulator for this thread's portion
    LocalTensor<float, 16> local_C(Shape({4, 4})); // Each thread handles 4x4
    local_C.view().fill(0.0f);
    
    // Main computation loop over K dimension
    for (index_t k_block = 0; k_block < K; k_block += TILE_K) {
        
        // Calculate global indices for this tile
        index_t global_row = block_row * TILE_M;
        index_t global_col = block_col * TILE_N;
        index_t global_k = k_block;
        
        // Create global memory tile views
        auto global_A_tile = A.slice(global_row, global_row + TILE_M)
                              .slice(global_k, global_k + TILE_K);
        auto global_B_tile = B.slice(global_k, global_k + TILE_K)
                              .slice(global_col, global_col + TILE_N);
        
        // Data movement: Global -> Shared
        // This is the key abstraction: dst = src.tile(shape)
        shared_A = global_A_tile.tile(Shape({TILE_M, TILE_K}));
        shared_B = global_B_tile.tile(Shape({TILE_K, TILE_N}));
        
        __syncthreads(); // Wait for all data to be loaded
        
        // Inner loop for tensor core operations
        for (dim_t k_inner = 0; k_inner < TILE_K; k_inner += 16) {
            
            // Create 16x16 subtiles for tensor cores
            auto A_subtile = shared_A.slice(thread_row * 16, (thread_row + 1) * 16)
                                    .slice(k_inner, k_inner + 16);
            auto B_subtile = shared_B.slice(k_inner, k_inner + 16)
                                    .slice(thread_col * 16, (thread_col + 1) * 16);
            
            // Check if we can use tensor cores
            if (can_use_tensor_cores() && A_subtile.shape().numel() == 256) {
                
                // Create a local result tile for tensor core output
                LocalTensor<float, 256> tc_result(Shape({16, 16}));
                
                // Tensor core computation: C += A * B
                tensor_core_mma_16x16x16(A_subtile, B_subtile, tc_result.view(), 
                                       alpha, beta);
                
                // Accumulate into local result
                auto local_view = local_C.view();
                for (dim_t i = 0; i < 16; ++i) {
                    for (dim_t j = 0; j < 16; ++j) {
                        if (i < 4 && j < 4) { // Thread handles 4x4 portion
                            local_view(i, j) += tc_result[i * 16 + j];
                        }
                    }
                }
                
            } else {
                // Fallback: Manual computation for non-tensor-core cases
                for (dim_t i = 0; i < 4; ++i) {
                    for (dim_t j = 0; j < 4; ++j) {
                        float sum = 0.0f;
                        for (dim_t k = 0; k < 16; ++k) {
                            sum += static_cast<float>(A_subtile(i, k)) * 
                                   static_cast<float>(B_subtile(k, j));
                        }
                        local_C[i * 4 + j] += alpha * sum;
                    }
                }
            }
        }
        
        __syncthreads(); // Synchronize before next iteration
    }
    
    // Write results back to global memory
    index_t global_c_row = block_row * TILE_M + thread_row * 4;
    index_t global_c_col = block_col * TILE_N + thread_col * 4;
    
    if (global_c_row < M && global_c_col < N) {
        auto global_C_tile = C.slice(global_c_row, global_c_row + 4)
                              .slice(global_c_col, global_c_col + 4);
        
        // Copy local results to global memory
        auto local_view = local_C.view();
        for (dim_t i = 0; i < 4; ++i) {
            for (dim_t j = 0; j < 4; ++j) {
                if (global_c_row + i < M && global_c_col + j < N) {
                    global_C_tile(i, j) = local_view(i, j) + 
                                         beta * global_C_tile(i, j);
                }
            }
        }
    }
}

/**
 * @brief Simpler matrix multiplication using shared memory tiling
 * 
 * This demonstrates a more straightforward tiling approach without tensor cores
 */
template<typename T, dim_t TILE_SIZE>
__global__ void gemm_shared_memory_kernel(
    TensorView<T> A, TensorView<T> B, TensorView<T> C,
    index_t M, index_t N, index_t K) {
    
    // Shared memory for tiles
    SharedTensor<T, TILE_SIZE * TILE_SIZE> shared_A(Shape({TILE_SIZE, TILE_SIZE}));
    SharedTensor<T, TILE_SIZE * TILE_SIZE> shared_B(Shape({TILE_SIZE, TILE_SIZE}));
    
    // Thread local accumulator
    T local_sum = T(0);
    
    // Thread indices within tile
    const dim_t tx = threadIdx.x;
    const dim_t ty = threadIdx.y;
    
    // Global position
    const index_t row = blockIdx.y * TILE_SIZE + ty;
    const index_t col = blockIdx.x * TILE_SIZE + tx;
    
    // Main loop over K dimension
    for (index_t k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        
        // Load tiles into shared memory
        if (row < M && k_tile + tx < K) {
            shared_A[ty * TILE_SIZE + tx] = A(row, k_tile + tx);
        } else {
            shared_A[ty * TILE_SIZE + tx] = T(0);
        }
        
        if (k_tile + ty < K && col < N) {
            shared_B[ty * TILE_SIZE + tx] = B(k_tile + ty, col);
        } else {
            shared_B[ty * TILE_SIZE + tx] = T(0);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (dim_t k = 0; k < TILE_SIZE; ++k) {
            local_sum += shared_A[ty * TILE_SIZE + k] * 
                        shared_B[k * TILE_SIZE + tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C(row, col) = local_sum;
    }
}

/**
 * @brief Convolution kernel using tensor abstractions
 * 
 * Demonstrates how to express convolution operations using tile() and tensor views
 */
template<typename T>
__global__ void conv2d_kernel(
    TensorView<T> input,   // (N, C, H, W)
    TensorView<T> weight,  // (K, C, R, S)  
    TensorView<T> output,  // (N, K, H_out, W_out)
    dim_t stride_h, dim_t stride_w,
    dim_t pad_h, dim_t pad_w) {
    
    // Get output position
    const index_t n = blockIdx.z;
    const index_t k = blockIdx.y;
    const index_t h_out = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t w_out = blockIdx.x * blockDim.y + threadIdx.y;
    
    auto output_shape = output.shape();
    if (h_out >= output_shape[2] || w_out >= output_shape[3]) return;
    
    // Local accumulator
    T sum = T(0);
    
    // Get weight tile for this output channel
    auto weight_tile = weight.slice(k, k + 1); // (1, C, R, S)
    
    // Convolution computation
    auto weight_shape = weight.shape();
    for (index_t c = 0; c < weight_shape[1]; ++c) {
        for (index_t r = 0; r < weight_shape[2]; ++r) {
            for (index_t s = 0; s < weight_shape[3]; ++s) {
                
                // Calculate input position
                index_t h_in = h_out * stride_h - pad_h + r;
                index_t w_in = w_out * stride_w - pad_w + s;
                
                auto input_shape = input.shape();
                if (h_in >= 0 && h_in < input_shape[2] && 
                    w_in >= 0 && w_in < input_shape[3]) {
                    
                    sum += input(n, c, h_in, w_in) * weight(k, c, r, s);
                }
            }
        }
    }
    
    // Write result
    output(n, k, h_out, w_out) = sum;
}

/**
 * @brief Host function to launch GEMM kernel
 */
template<typename T>
void launch_gemm_kernel(const Tensor<T>& A, const Tensor<T>& B, Tensor<float>& C,
                       float alpha = 1.0f, float beta = 0.0f) {
    
    auto A_shape = A.shape();
    auto B_shape = B.shape();
    auto C_shape = C.shape();
    
    index_t M = A_shape[0];
    index_t K = A_shape[1]; 
    index_t N = B_shape[1];
    
    // Kernel launch configuration
    const dim_t TILE_M = 128;
    const dim_t TILE_N = 128; 
    const dim_t TILE_K = 32;
    
    dim3 block_size(TILE_N / 16, TILE_M / 16); // 16x16 threads per block
    dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    // Launch kernel with tensor views
    gemm_tensor_core_kernel<T, TILE_M, TILE_N, TILE_K><<<grid_size, block_size>>>(
        A.view(), B.view(), C.view(), M, N, K, alpha, beta
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template void launch_gemm_kernel<float>(const Tensor<float>&, const Tensor<float>&, 
                                       Tensor<float>&, float, float);
template void launch_gemm_kernel<__half>(const Tensor<__half>&, const Tensor<__half>&, 
                                        Tensor<float>&, float, float); 