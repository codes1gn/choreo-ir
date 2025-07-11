# Choreo-IR Build Scripts

## Quick Start

在项目根目录运行：

```bash
./scripts/build.sh
```

这个脚本会自动：
- 配置CUDA 12.6环境
- 检测GPU架构（当前支持A100自动配置为arch 80）
- 编译所有核心测试
- 运行基础验证测试

## 系统要求

- CUDA 12.6已安装在 `/usr/local/cuda-12.6`
- GCC 8.x 或 9.x（脚本会自动选择）
- CMake 3.16+
- GoogleTest库

## 输出

成功运行后，你会看到：
- ✅ 绿色的成功消息
- ⚠️ 黄色的警告（非致命）
- ❌ 红色的错误（会停止脚本）

编译产物在 `build/` 目录：
- `tests/test_end2end` - 端到端测试
- `tests/test_device` - 设备测试
- `tests/test_shape` - 形状测试
- `tests/test_stride` - 步长测试
- `tests/test_layout` - 布局测试
- `tests/test_tensor` - 张量测试

## 手动测试

```bash
cd build
./tests/test_end2end
./tests/test_device
./tests/test_shape
```

## 已知问题

1. cuBLAS参数错误（Error code: 7/13）- 这是API使用问题，不是C++17迁移问题
2. `test_performance_comparison` 需要 `choreo-ir-baselines` 库（已跳过）

## C++17特性

项目现在使用：
- `std::is_same_v<T, U>` 替代 `IS_SAME(T, U)` 宏
- `if constexpr` 用于编译时条件判断
- C++17标准库特性

所有模板元编程都使用了C++17的编译时特性。 