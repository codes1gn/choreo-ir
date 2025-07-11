/**
 * @file cuda_baseline.cu
 * @brief CUDA kernel implementations for baseline
 */

#include "cuda_baseline.hpp"

namespace choreo_ir {
namespace baselines {

// CUDA kernel for matrix multiplication (row-major)
template<typename T>
__global__ void gemm_kernel_row(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T sum = static_cast<T>(0);
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
// CUDA kernel for matrix multiplication (col-major)
template<typename T>
__global__ void gemm_kernel_col(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        T sum = static_cast<T>(0);
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[col * K + k];
        }
        C[row * N + col] = sum;  // Output in row-major format
    }
}

// CUDA kernel for vector addition
template<typename T>
__global__ void axpy_kernel(int n, const T* alpha, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = *alpha * x[idx] + y[idx];
    }
}

// CUDA kernel for vector copy
template<typename T>
__global__ void copy_kernel(int n, const T* x, T* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx];
    }
}

// Implementation functions that call the kernels
template<typename T>
void gemm_kernel_impl(const T* A, const T* B, T* C, int M, int N, int K, dim3 gridDim, dim3 blockDim, DataLayout layout) {
    if (layout == DataLayout::ROW_MAJOR) {
        gemm_kernel_row<T><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    } else {
        gemm_kernel_col<T><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
}

template<typename T>
void axpy_kernel_impl(int n, const T* alpha, const T* x, T* y, 
                     int gridSize, int blockSize) {
    axpy_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, y);
}

template<typename T>
void copy_kernel_impl(int n, const T* x, T* y, 
                     int gridSize, int blockSize) {
    copy_kernel<T><<<gridSize, blockSize>>>(n, x, y);
}

// Explicit template instantiations
template void gemm_kernel_impl<float>(const float*, const float*, float*, 
                                     int, int, int, dim3, dim3, DataLayout);
template void gemm_kernel_impl<double>(const double*, const double*, double*, 
                                      int, int, int, dim3, dim3, DataLayout);
template void gemm_kernel_impl<__half>(const __half*, const __half*, __half*, 
                                       int, int, int, dim3, dim3, DataLayout);

template void axpy_kernel_impl<float>(int, const float*, const float*, float*, 
                                     int, int);
template void axpy_kernel_impl<double>(int, const double*, const double*, double*, 
                                      int, int);
template void axpy_kernel_impl<__half>(int, const __half*, const __half*, __half*, 
                                       int, int);

template void copy_kernel_impl<float>(int, const float*, float*, int, int);
template void copy_kernel_impl<double>(int, const double*, double*, int, int);
template void copy_kernel_impl<__half>(int, const __half*, __half*, int, int);

} // namespace baselines
} // namespace choreo_ir 