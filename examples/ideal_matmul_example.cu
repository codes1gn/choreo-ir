/**
 * @file ideal_matmul_example.cu
 * @brief Ideal example of matrix multiplication using Choreo-IR abstractions
 * 
 * This demonstrates the most intuitive programming model:
 * 1. Natural tensor declarations: GlobalTensor<T>, SharedTensor<T>, LocalTensor<T>
 * 2. Intuitive data movement: shared_A = global_A.tile(shape)
 * 3. Simple tensor core usage: local_C = local_A * local_B
 * 4. Automatic launch configuration based on tensor shapes and layouts
 */

#include <cuda_runtime.h>
#include "choreo-ir/choreo-ir.hpp"

using namespace choreo_ir;

/**
 * @brief Ideal matrix multiplication kernel using natural tensor abstractions
 * 
 * Programming pattern:
 * 1. Declare tensors with their memory space
 * 2. Use assignment for data movement between memory hierarchies
 * 3. Use * operator for tensor core operations
 * 4. Framework automatically handles coalescing, padding, etc.
 */
template<typename T, int TILE_M = 128, int TILE_N = 128, int TILE_K = 32>
__global__ void ideal_matmul_kernel(
    GlobalTensor<T> A,     // Input matrix A in global memory
    GlobalTensor<T> B,     // Input matrix B in global memory  
    GlobalTensor<T> C      // Output matrix C in global memory
) {
    // Get matrix dimensions from tensors
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    // Block-level shared memory tensors
    SharedTensor<T, TILE_M, TILE_K> shared_A;
    SharedTensor<T, TILE_K, TILE_N> shared_B;
    
    // Thread-local tensors for tensor cores (16x16 fragments)
    LocalTensor<T, 16, 16> local_A;
    LocalTensor<T, 16, 16> local_B;
    LocalTensor<float, 16, 16> local_C; // Accumulator in float
    
    // Initialize accumulator
    local_C.fill(0.0f);
    
    // Calculate this block's position
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    
    // Main computation loop over K dimension
    for (int k_offset = 0; k_offset < K; k_offset += TILE_K) {
        
        // Data movement: Global -> Shared
        // Framework automatically handles coalescing, boundary checks, etc.
        shared_A = A.tile({block_row, k_offset}, {TILE_M, TILE_K});
        shared_B = B.tile({k_offset, block_col}, {TILE_K, TILE_N});
        
        __syncthreads(); // Synchronize shared memory loads
        
        // Inner loop for tensor core operations
        for (int k_inner = 0; k_inner < TILE_K; k_inner += 16) {
            
            // Data movement: Shared -> Local (registers)
            // Framework maps thread to appropriate 16x16 tile
            local_A = shared_A.subtile(k_inner);
            local_B = shared_B.subtile(k_inner);
            
            // Tensor core computation: C += A * B
            // Framework automatically selects WMMA/MMA based on hardware
            local_C = local_C + (local_A * local_B);
        }
        
        __syncthreads(); // Synchronize before next iteration
    }
    
    // Write results back: Local -> Global
    // Framework handles the accumulation and boundary checks
    C.accumulate(local_C, {block_row, block_col});
}

/**
 * @brief Host function with automatic launch configuration
 * 
 * The framework analyzes tensor shapes, layouts, and operations to automatically
 * determine optimal grid/block dimensions, shared memory usage, etc.
 */
template<typename T>
void matmul(const HostTensor<T>& A, const HostTensor<T>& B, HostTensor<T>& C) {
    
    // Get matrix dimensions
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    assert(K == K2 && "Matrix dimensions must match");
    assert(C.shape() == std::make_tuple(M, N) && "Output shape must match");
    
    // Create device tensors (automatic memory management)
    auto d_A = A.to_device();
    auto d_B = B.to_device(); 
    auto d_C = GlobalTensor<T>::zeros({M, N});
    
    // Framework automatically analyzes:
    // 1. Tensor shapes and determine optimal tile sizes
    // 2. Memory access patterns for coalescing
    // 3. Register usage for occupancy
    // 4. Shared memory requirements
    // 5. Hardware capabilities (tensor core support)
    auto config = AutoConfig::analyze(d_A, d_B, d_C);
    
    // Launch kernel with auto-generated configuration
    config.launch(ideal_matmul_kernel<T>, d_A, d_B, d_C);
    
    // Copy result back to host
    C = d_C.to_host();
}

/**
 * @brief Simplified user API - looks like a normal function call
 */
template<typename T>
HostTensor<T> operator*(const HostTensor<T>& A, const HostTensor<T>& B) {
    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();
    
    HostTensor<T> C({M, N});
    matmul(A, B, C);
    return C;
}

/**
 * @brief Example usage from user perspective
 */
void user_example() {
    // Create matrices on host
    auto A = HostTensor<float>::random({1024, 512});
    auto B = HostTensor<float>::random({512, 1024});
    
    // Matrix multiplication - looks like normal math!
    auto C = A * B;
    
    // Or explicit call
    auto C2 = HostTensor<float>({1024, 1024});
    matmul(A, B, C2);
    
    std::cout << "Result shape: " << C.shape() << std::endl;
}

/**
 * @brief Advanced example with mixed precision
 */
void mixed_precision_example() {
    // Input in half precision
    auto A = HostTensor<__half>::random({2048, 1024});
    auto B = HostTensor<__half>::random({1024, 2048});
    
    // Computation automatically uses tensor cores with half->float accumulation
    auto C = HostTensor<float>({2048, 2048});
    matmul(A, B, C); // Framework handles type promotion automatically
}

/**
 * @brief Example showing different tensor core shapes
 */
template<typename T>
__global__ void flexible_tensor_core_kernel(
    GlobalTensor<T> A, GlobalTensor<T> B, GlobalTensor<float> C
) {
    // Different tensor core configurations based on data type and hardware
    if constexpr (std::is_same_v<T, __half>) {
        // For half precision, use 16x16x16 tiles
        LocalTensor<T, 16, 16> local_A, local_B;
        LocalTensor<float, 16, 16> local_C;
        
        // Framework selects WMMA for half precision
        local_C = local_A * local_B; // Uses __half tensor cores
        
    } else if constexpr (std::is_same_v<T, float>) {
        // For float, use different strategy or fallback to CUDA cores
        LocalTensor<T, 8, 8> local_A, local_B;
        LocalTensor<T, 8, 8> local_C;
        
        // Manual computation for float (no tensor core support)
        local_C = local_A.manual_multiply(local_B);
    }
}

/**
 * @brief Convolution example using the same abstractions
 */
template<typename T>
__global__ void ideal_conv2d_kernel(
    GlobalTensor<T> input,    // (N, C, H, W)
    GlobalTensor<T> weight,   // (K, C, R, S)
    GlobalTensor<T> output    // (N, K, H_out, W_out)
) {
    // Get convolution parameters from tensor shapes
    auto [N, C, H, W] = input.shape();
    auto [K, C2, R, S] = weight.shape();
    
    // Shared memory for input and weight tiles
    SharedTensor<T, 64, 64> shared_input_tile;
    SharedTensor<T, 64, 64> shared_weight_tile;
    
    // Local tensors for computation
    LocalTensor<T, 16, 16> local_input;
    LocalTensor<T, 16, 16> local_weight;
    LocalTensor<float, 16, 16> local_output;
    
    // Calculate output position
    auto [out_n, out_k, out_h, out_w] = get_output_position();
    
    // Convolution loop
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                
                // Calculate input position
                int in_h = out_h + r;
                int in_w = out_w + s;
                
                // Load tiles
                shared_input_tile = input.tile({out_n, c, in_h, in_w}, {1, 1, 16, 16});
                shared_weight_tile = weight.tile({out_k, c, r, s}, {1, 1, 1, 1});
                
                // Move to local memory and compute
                local_input = shared_input_tile.to_local();
                local_weight = shared_weight_tile.to_local();
                
                // Accumulate using tensor cores if possible
                local_output = local_output + (local_input * local_weight);
            }
        }
    }
    
    // Write result
    output.write(local_output, {out_n, out_k, out_h, out_w});
}

/**
 * @brief Framework's auto-configuration system (conceptual)
 */
class AutoConfig {
public:
    template<typename... TensorTypes>
    static LaunchConfig analyze(const TensorTypes&... tensors) {
        LaunchConfig config;
        
        // Analyze tensor shapes and access patterns
        auto shapes = std::make_tuple(tensors.shape()...);
        auto layouts = std::make_tuple(tensors.layout()...);
        
        // Determine optimal tile sizes based on:
        // 1. Tensor core requirements (16x16, 8x32, etc.)
        // 2. Shared memory constraints
        // 3. Register usage
        // 4. Memory coalescing patterns
        config.tile_sizes = optimize_tile_sizes(shapes, layouts);
        
        // Calculate grid and block dimensions
        config.grid_dim = calculate_grid_dimensions(shapes, config.tile_sizes);
        config.block_dim = calculate_block_dimensions(config.tile_sizes);
        
        // Estimate resource usage
        config.shared_memory = estimate_shared_memory_usage(config.tile_sizes);
        config.register_usage = estimate_register_usage(config.tile_sizes);
        
        // Hardware-specific optimizations
        if (has_tensor_cores()) {
            config.use_tensor_cores = true;
            config.tensor_core_type = select_optimal_tensor_core_type();
        }
        
        return config;
    }
    
    template<typename KernelFunc, typename... Args>
    void launch(KernelFunc kernel, Args&&... args) const {
        // Set shared memory if needed
        if (shared_memory > 0) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
        }
        
        // Launch kernel with calculated configuration
        kernel<<<grid_dim, block_dim, shared_memory>>>(std::forward<Args>(args)...);
        
        // Check for errors
        CUDA_CHECK(cudaGetLastError());
    }
    
private:
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_memory = 0;
    size_t register_usage = 0;
    bool use_tensor_cores = false;
    TensorCoreType tensor_core_type = TensorCoreType::AUTO;
    
    // Private helper methods would be implemented here...
}; 