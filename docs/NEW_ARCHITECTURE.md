# Choreo-IR 新架构设计

## 核心理念

经过重构，Choreo-IR 现在是一个**编程抽象库**而不是算子实现库。其核心目标是：

1. **简化 CUDA kernel 编程**：通过类似 `dst = src.tile(shape)` 的语法表达数据搬运
2. **零成本抽象**：编译时优化，运行时无额外开销
3. **自动化配置**：根据 tensor 形状和硬件自动生成最优 launch 配置
4. **自然的 tensor core 调用**：`local_C = local_A * local_B`

## 架构概览

```
choreo-ir/
├── include/choreo-ir/           # Header-only 库核心
│   ├── core/                    # 基础类型和配置
│   ├── tensor/                  # Tensor 抽象和内存层次
│   ├── compute/                 # Tensor core 指令抽象
│   └── utils/                   # 工具函数
├── benchmark/                   # 算子实现和性能测试
│   ├── kernels/                 # 使用抽象编写的 CUDA kernels
│   ├── baselines/               # cuBLAS/cuDNN 基准实现
│   └── comparisons/             # 性能比较
└── examples/                    # 使用示例
```

## 编程模型

### 1. Tensor 声明

```cpp
// 不同内存层次的 tensor 声明
global tensor<half> A;           // 全局内存
shared tensor<half> shared_A;    // 共享内存  
local tensor<half> local_A;      // 寄存器/本地内存
```

### 2. 数据传输

```cpp
// 直观的数据搬运语法
shared_A = global_A.tile({row, col}, {TILE_M, TILE_K});  // Global -> Shared
local_A = shared_A.tile(offset, {16, 16});            // Shared -> Local
```

### 3. Tensor Core 操作

```cpp
// 自然的矩阵乘法语法
local_C = local_A * local_B;     // 自动选择 WMMA/MMA 指令
local_C += local_A * local_B;    // 累加操作
```

### 4. 自动配置

```cpp
// 用户无需关心 launch 配置
auto config = auto_analyze(A, B, C);
config.launch(my_kernel, A, B, C);
```

## 核心组件

### 1. 内存层次抽象

```cpp
enum class MemorySpace {
    GLOBAL,    // 全局设备内存
    SHARED,    // 块级共享内存
    LOCAL,     // 线程本地内存（寄存器）
    TEXTURE,   // 纹理内存
    CONSTANT   // 常量内存
};
```

### 2. Tensor 类型系统

```cpp
template<typename T>
class TensorView {
    T* data_;
    Layout layout_;
    MemorySpace memory_space_;
    
    // 数据传输操作
    TensorView copy_to_shared(T* shared_ptr, const Shape& tile_shape);
    TensorView copy_to_local(T (&local_array)[SIZE]);
    
    // 赋值运算符实现数据搬运
    template<typename SrcT>
    TensorView& operator=(const TensorView<SrcT>& src);
};
```

### 3. Tensor Core 抽象

```cpp
// WMMA fragment 包装
template<typename T, int M, int N, int K>
class WmmaFragment;

// Tensor core 操作
template<typename T>
__device__ void tensor_core_mma_16x16x16(
    const TensorView<T>& A, 
    const TensorView<T>& B,
    TensorView<float>& C
);
```

### 4. 自动配置系统

```cpp
class AutoConfig {
    template<typename... Tensors>
    static LaunchConfig analyze(const Tensors&... tensors);
    
    template<typename KernelFunc, typename... Args>
    void launch(KernelFunc kernel, Args&&... args) const;
};
```

## 使用示例

### 理想的矩阵乘法 Kernel

```cpp
template<typename T>
__global__ void simple_matmul_kernel(
    global tensor<T> A,
    global tensor<T> B,  
    global tensor<T> C
) {
    // 声明 shared 和 local tensors
    shared tensor<T> shared_A, shared_B;
    local tensor<T> local_A, local_B;
    local tensor<float> local_C;
    
    local_C.fill(0.0f);
    
    // 主循环
    for (auto k_tile : range(A.shape()[1], tile_size_k)) {
        // 数据传输：Global -> Shared
        shared_A = A.tile({block_row, k_tile}, {tile_size_m, tile_size_k});
        shared_B = B.tile({k_tile, block_col}, {tile_size_k, tile_size_n});
        
        sync_threads();
        
        // 内层循环：Shared -> Local -> Tensor Core
        for (auto k_inner : range(tile_size_k, 16)) {
            local_A = shared_A.tile(k_inner, {16, 16});
            local_B = shared_B.tile(k_inner, {16, 16});
            
            // Tensor core 计算
            local_C += local_A * local_B;
        }
        
        sync_threads();
    }
    
    // 写回结果
    C.accumulate(local_C, {block_row, block_col});
}
```

### 用户 API

```cpp
// 用户只需要这样调用
auto A = host_tensor<half>::random({1024, 512});
auto B = host_tensor<half>::random({512, 1024});

// 矩阵乘法 - 就像数学运算！
auto C = A * B;

// 或者显式调用
auto C2 = host_tensor<float>({1024, 1024});
matmul(A, B, C2);
```

## 自动配置规则

框架根据以下因素自动生成最优配置：

1. **Tensor 形状**：确定最优 tile 大小
2. **数据类型**：选择合适的 tensor core 指令
3. **硬件能力**：根据 compute capability 选择 WMMA/MMA
4. **内存访问模式**：优化 coalescing 和 bank conflicts
5. **资源约束**：平衡 shared memory、registers、occupancy

### 配置示例

```cpp
// 对于 half 精度 1024x1024 矩阵乘法
auto config = AutoConfig::analyze(A, B, C);

// 自动生成的配置可能是：
// - tile_size: 128x128x32
// - block_dim: (8, 16)  // 8x16 = 128 threads
// - grid_dim: (8, 8)    // 处理 1024/128 = 8 个 tiles
// - shared_memory: 96KB
// - use_tensor_cores: true
// - tensor_core_type: WMMA_16x16x16
```

## 性能优化策略

1. **编译时优化**：基于 tensor 形状和类型的模板特化
2. **内存层次优化**：自动选择最优的数据搬运策略
3. **指令选择**：根据硬件自动选择 WMMA/MMA/CUDA cores
4. **占用率优化**：平衡 register 使用和 occupancy
5. **访存优化**：自动处理 coalescing 和 padding

## 与现有库的比较

| 特性 | Choreo-IR | cuBLAS | cutlass | PyTorch |
|------|-----------|--------|---------|---------|
| 编程模型 | 直观的 tensor 操作 | 函数调用 | C++ 模板 | Python API |
| 性能 | 零成本抽象 | 高度优化 | 高度优化 | Python 开销 |
| 可扩展性 | 用户可编写 kernel | 固定 API | 可扩展 | 有限 |
| 学习曲线 | 接近数学表达 | BLAS API | 复杂模板 | 简单 |
| 自动优化 | 自动配置 | 内置优化 | 手动调优 | 自动 |

## 未来扩展

1. **更多 tensor core 支持**：支持 Hopper 架构的新指令
2. **自动调优**：基于运行时 profiling 的配置优化
3. **多 GPU 支持**：跨 GPU 的 tensor 操作
4. **融合操作**：自动融合多个 tensor 操作
5. **动态形状**：支持运行时变化的 tensor 形状

## 总结

新的 Choreo-IR 架构提供了：

- **直观的编程模型**：`dst = src.tile(shape)` 和 `C = A * B`
- **零成本抽象**：编译时优化，无运行时开销
- **自动化配置**：用户无需关心 CUDA 实现细节
- **高性能**：与手写 CUDA 代码性能相当
- **可扩展性**：用户可以轻松编写新的 kernel

这使得 CUDA 编程变得更像写数学公式，同时保持了高性能计算的能力。 