/**
 * @file types.hpp
 * @brief Core type definitions for the Choreo-IR library
 */

#ifndef CHOREO_IR_CORE_TYPES_HPP
#define CHOREO_IR_CORE_TYPES_HPP

#include <cstdint>
#include <cuda_fp16.h>

// Conditional inclusion for bfloat16 support
#if defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#define CHOREO_IR_HAS_BF16 1
#else
#define CHOREO_IR_HAS_BF16 0
#endif

namespace choreo_ir {

// Basic integer types
using int8_t = std::int8_t;
using int16_t = std::int16_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;
using uint8_t = std::uint8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;

// Floating point types
using float32_t = float;
using float64_t = double;
using float16_t = __half;

#if CHOREO_IR_HAS_BF16
using bfloat16_t = __nv_bfloat16;
#else
// Fallback for older CUDA versions
struct bfloat16_t {
    uint16_t x;
    bfloat16_t() = default;
    bfloat16_t(uint16_t val) : x(val) {}
    operator uint16_t() const { return x; }
};
#endif

// Index and dimension types
using index_t = int64_t;
using dim_t = int32_t;
using size_t = std::size_t;

/**
 * @enum DataType
 * @brief Enumeration of supported data types
 */
enum class DataType : uint8_t {
    INT8 = 0,
    INT16 = 1,
    INT32 = 2,
    INT64 = 3,
    UINT8 = 4,
    UINT16 = 5,
    UINT32 = 6,
    UINT64 = 7,
    FLOAT16 = 8,
    FLOAT32 = 9,
    FLOAT64 = 10,
    BFLOAT16 = 11
};

/**
 * @brief Get the size in bytes of a data type
 * @param dtype The data type
 * @return Size in bytes
 */
constexpr size_t sizeof_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::UINT8:
            return 1;
        case DataType::INT16:
        case DataType::UINT16:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::FLOAT64:
            return 8;
        default:
            return 0;
    }
}

/**
 * @brief Template to get DataType from C++ type
 */
template<typename T>
struct type_to_dtype;

template<> struct type_to_dtype<int8_t> { static constexpr DataType value = DataType::INT8; };
template<> struct type_to_dtype<int16_t> { static constexpr DataType value = DataType::INT16; };
template<> struct type_to_dtype<int32_t> { static constexpr DataType value = DataType::INT32; };
template<> struct type_to_dtype<int64_t> { static constexpr DataType value = DataType::INT64; };
template<> struct type_to_dtype<uint8_t> { static constexpr DataType value = DataType::UINT8; };
template<> struct type_to_dtype<uint16_t> { static constexpr DataType value = DataType::UINT16; };
template<> struct type_to_dtype<uint32_t> { static constexpr DataType value = DataType::UINT32; };
template<> struct type_to_dtype<uint64_t> { static constexpr DataType value = DataType::UINT64; };
template<> struct type_to_dtype<float16_t> { static constexpr DataType value = DataType::FLOAT16; };
template<> struct type_to_dtype<float32_t> { static constexpr DataType value = DataType::FLOAT32; };
template<> struct type_to_dtype<float64_t> { static constexpr DataType value = DataType::FLOAT64; };
template<> struct type_to_dtype<bfloat16_t> { static constexpr DataType value = DataType::BFLOAT16; };

/**
 * @brief Get DataType from template type
 */
template<typename T>
constexpr DataType get_dtype() {
    return type_to_dtype<T>::value;
}

/**
 * @enum MemoryType
 * @brief Memory location types
 */
enum class MemoryType : uint8_t {
    HOST = 0,
    DEVICE = 1,
    UNIFIED = 2
};

/**
 * @enum ComputeCapability
 * @brief CUDA compute capability versions
 */
enum class ComputeCapability : uint8_t {
    SM_70 = 70,  // V100
    SM_75 = 75,  // T4, RTX 20xx
    SM_80 = 80,  // A100
    SM_86 = 86,  // RTX 30xx
    SM_89 = 89,  // RTX 40xx
    SM_90 = 90   // H100
};

} // namespace choreo_ir

#endif // CHOREO_IR_CORE_TYPES_HPP 