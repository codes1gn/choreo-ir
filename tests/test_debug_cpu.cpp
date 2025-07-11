#include <iostream>
#include <vector>
#include "choreo-ir/choreo-ir.hpp"

using namespace choreo_ir;

int main() {
    // Initialize
    if (!initialize()) {
        std::cerr << "Failed to initialize" << std::endl;
        return 1;
    }
    
    // Create simple test case using HOST memory (CPU)
    const int M = 2, N = 2, K = 2;
    
    std::cout << "Creating HOST tensors..." << std::endl;
    
    // Create simple matrices with known values on HOST
    auto A = Tensor<float>::zeros(Shape({M, K}), MemoryType::HOST);
    auto B = Tensor<float>::zeros(Shape({K, N}), MemoryType::HOST);
    
    // Set A = [[1, 2], [3, 4]]
    std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f};
    A.copy_from_host(A_data.data());
    
    // Set B = [[1, 0], [0, 1]] (identity)
    std::vector<float> B_data = {1.0f, 0.0f, 0.0f, 1.0f};
    B.copy_from_host(B_data.data());
    
    std::cout << "Input A (HOST):" << std::endl;
    std::vector<float> A_host(M * K);
    A.copy_to_host(A_host.data());
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << A_host[i * K + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Input B (HOST):" << std::endl;
    std::vector<float> B_host(K * N);
    B.copy_to_host(B_host.data());
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B_host[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Perform multiplication (this should use CPU implementation)
    std::cout << "Performing A * B (CPU)..." << std::endl;
    auto C = A * B;
    
    std::cout << "Result C (CPU):" << std::endl;
    std::vector<float> C_host(M * N);
    C.copy_to_host(C_host.data());
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C_host[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Expected result should be A * I = A
    // C should be [[1, 2], [3, 4]]
    
    finalize();
    return 0;
} 