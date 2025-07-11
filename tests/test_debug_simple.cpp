#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // Initialize
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int M = 2, N = 2, K = 2;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f}; // row-major: [[1,2],[3,4]]
    std::vector<float> h_B = {1.0f, 0.0f, 0.0f, 1.0f}; // row-major: identity
    std::vector<float> h_C(4, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    // 行主序模拟：C = A * B
    // cuBLAS: C^T = B^T * A^T
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_B, K, // B^T: (N,K), lda=K
        d_A, M, // A^T: (K,M), lda=M
        &beta,
        d_C, N // C^T: (N,M), lda=N
    );

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Result C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaDeviceReset();
    return 0;
} 