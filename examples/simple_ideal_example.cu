/**
 * @file simple_ideal_example.cu
 * @brief 最简化的理想抽象示例
 * 
 * 展示核心编程模式：
 * 1. 直观的tensor声明：global/shared/local
 * 2. 自然的数据传输：dst = src.tile(shape)
 * 3. 简单的tensor core调用：C = A * B
 * 4. 自动配置管理
 */

#include "choreo-ir/ideal.hpp"

using namespace choreo_ir;

/**
 * @brief 理想的矩阵乘法kernel
 * 用户只需要关心算法逻辑，不需要考虑具体的CUDA实现细节
 */
template<typename T>
__global__ void simple_matmul_kernel(
    global tensor<T> A,    // 全局内存中的矩阵A
    global tensor<T> B,    // 全局内存中的矩阵B  
    global tensor<T> C     // 全局内存中的矩阵C
) {
    // 声明shared memory tensors
    shared tensor<T> shared_A;
    shared tensor<T> shared_B;
    
    // 声明local tensors (registers)
    local tensor<T> local_A;
    local tensor<T> local_B;
    local tensor<float> local_C; // accumulator使用float
    
    // 初始化累加器
    local_C.fill(0.0f);
    
    // 获取当前block负责的tile位置
    auto [block_row, block_col] = get_block_position();
    
    // 主循环：遍历K维度
    for (auto k_tile : range(A.shape()[1], tile_size_k)) {
        
        // 数据传输：Global -> Shared
        // 框架自动处理内存对齐、边界检查、coalescing等
        shared_A = A.tile({block_row, k_tile}, {tile_size_m, tile_size_k});
        shared_B = B.tile({k_tile, block_col}, {tile_size_k, tile_size_n});
        
        sync_threads(); // 等待shared memory加载完成
        
        // 内层循环：tensor core操作
        for (auto k_inner : range(tile_size_k, 16)) {
            
            // 数据传输：Shared -> Local
            // 框架自动映射线程到合适的16x16 tile
            local_A = shared_A.subtile(k_inner, {16, 16});
            local_B = shared_B.subtile(k_inner, {16, 16});
            
            // Tensor core计算：C += A * B
            // 框架根据硬件自动选择WMMA/MMA指令
            local_C += local_A * local_B;
        }
        
        sync_threads(); // 下一次迭代前同步
    }
    
    // 写回结果：Local -> Global
    // 框架处理累加和边界检查
    C.accumulate(local_C, {block_row, block_col});
}

/**
 * @brief 用户API：就像调用普通函数一样
 */
template<typename T>
void matmul(const host_tensor<T>& A, const host_tensor<T>& B, host_tensor<T>& C) {
    
    // 自动配置分析
    // 框架分析tensor形状、layout、硬件能力，自动生成最优配置
    auto config = auto_analyze(A, B, C);
    
    // 自动启动kernel
    config.launch(simple_matmul_kernel<T>, A.to_device(), B.to_device(), C.to_device());
}

/**
 * @brief 最简单的用户调用
 */
void user_code() {
    // 创建矩阵
    auto A = host_tensor<half>::random({1024, 512});
    auto B = host_tensor<half>::random({512, 1024});
    
    // 矩阵乘法 - 看起来就像数学运算！
    auto C = A * B;
    
    // 或者显式调用
    auto C2 = host_tensor<float>({1024, 1024});
    matmul(A, B, C2);
}

/**
 * @brief Convolution示例：使用相同的抽象
 */
template<typename T>
__global__ void simple_conv2d_kernel(
    global tensor<T> input,    // (N, C, H, W)
    global tensor<T> weight,   // (K, C, R, S)
    global tensor<T> output    // (N, K, H_out, W_out)
) {
    // 声明shared和local tensors
    shared tensor<T> shared_input, shared_weight;
    local tensor<T> local_input, local_weight;
    local tensor<float> local_output;
    
    // 获取输出位置
    auto [n, k, h, w] = get_output_position();
    
    // 卷积计算
    for (auto [c, r, s] : product(weight.shape()[1], weight.shape()[2], weight.shape()[3])) {
        
        // 计算输入位置
        auto in_h = h * stride_h - pad_h + r;
        auto in_w = w * stride_w - pad_w + s;
        
        // 加载数据
        shared_input = input.tile({n, c, in_h, in_w}, {1, 1, tile_h, tile_w});
        shared_weight = weight.tile({k, c, r, s}, {1, 1, 1, 1});
        
        // 传输到local memory
        local_input = shared_input.to_local();
        local_weight = shared_weight.to_local();
        
        // 使用tensor cores（如果可能）
        local_output += local_input * local_weight;
    }
    
    // 写回结果
    output.write(local_output, {n, k, h, w});
}

/**
 * @brief 自动配置系统的核心概念
 */
struct AutoConfig {
    
    // 基于tensor形状和操作自动确定配置
    template<typename... Tensors>
    static LaunchConfig analyze(const Tensors&... tensors) {
        
        // 1. 分析tensor形状和access pattern
        auto shapes = extract_shapes(tensors...);
        auto patterns = analyze_access_patterns(tensors...);
        
        // 2. 根据硬件能力确定tile sizes
        auto tile_sizes = optimize_for_hardware(shapes, patterns);
        
        // 3. 计算grid/block dimensions
        auto [grid, block] = calculate_launch_dims(shapes, tile_sizes);
        
        // 4. 估算资源使用
        auto resources = estimate_resources(tile_sizes, patterns);
        
        return LaunchConfig{grid, block, resources.shared_mem, tile_sizes};
    }
    
    // 启动kernel的通用接口
    template<typename KernelFunc, typename... Args>
    void launch(KernelFunc kernel, Args&&... args) const {
        // 设置shared memory大小
        set_dynamic_shared_memory(kernel, shared_memory_size);
        
        // 启动kernel
        kernel<<<grid_dim, block_dim, shared_memory_size>>>(
            std::forward<Args>(args)...
        );
        
        check_kernel_launch();
    }
};

/**
 * @brief 编译时优化：基于tensor类型和形状的特化
 */
template<typename T, int M, int N, int K>
struct SpecializedConfig {
    static constexpr bool use_tensor_cores = 
        std::is_same_v<T, __half> && (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
    
    static constexpr int tile_m = use_tensor_cores ? 128 : 64;
    static constexpr int tile_n = use_tensor_cores ? 128 : 64;
    static constexpr int tile_k = use_tensor_cores ? 32 : 16;
    
    static constexpr TensorCoreType tc_type = 
        (__CUDA_ARCH__ >= 800) ? TensorCoreType::MMA_16x8x16 : TensorCoreType::WMMA_16x16x16;
};

/**
 * @brief 用户只看到的简单接口
 */
namespace user_api {
    
    // 矩阵乘法
    template<typename T>
    auto matmul(const host_tensor<T>& A, const host_tensor<T>& B) {
        return A * B; // 就这么简单！
    }
    
    // 卷积
    template<typename T>
    auto conv2d(const host_tensor<T>& input, const host_tensor<T>& weight, 
                int stride = 1, int padding = 0) {
        auto config = auto_analyze_conv2d(input, weight, stride, padding);
        return config.launch_and_return(simple_conv2d_kernel<T>, input, weight);
    }
    
    // 批量矩阵乘法
    template<typename T>
    auto batch_matmul(const host_tensor<T>& A, const host_tensor<T>& B) {
        // 框架自动处理batch维度
        return A.batch_multiply(B);
    }
}

/**
 * @brief 展示用户实际使用的代码
 */
void real_user_example() {
    using namespace user_api;
    
    // 就像使用NumPy/PyTorch一样简单！
    auto A = host_tensor<half>::random({2048, 1024});
    auto B = host_tensor<half>::random({1024, 2048});
    
    // 矩阵乘法
    auto C = matmul(A, B);
    
    // 卷积
    auto input = host_tensor<half>::random({32, 64, 128, 128});  // (N,C,H,W)
    auto weight = host_tensor<half>::random({128, 64, 3, 3});   // (K,C,R,S)
    auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
    
    // 批量操作
    auto batch_A = host_tensor<half>::random({16, 512, 256});
    auto batch_B = host_tensor<half>::random({16, 256, 512});
    auto batch_C = batch_matmul(batch_A, batch_B);
    
    std::cout << "结果形状: " << C.shape() << std::endl;
} 