/**
 * @file matmul_comparison.cpp
 * @brief Matrix multiplication performance comparison benchmark
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <memory>

#include "choreo-ir/choreo-ir.hpp"

// Include kernel implementations
extern "C" {
    // Our kernel implementations
    void matmul_naive_kernel_float(const float* A, const float* B, float* C,
                                  int64_t M, int64_t N, int64_t K,
                                  int64_t lda, int64_t ldb, int64_t ldc,
                                  float alpha, float beta);
    
    void matmul_shared_kernel_float(const float* A, const float* B, float* C,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t lda, int64_t ldb, int64_t ldc,
                                   float alpha, float beta);
    
    // cuBLAS implementations
    void cublas_matmul_impl_float(const choreo_ir::TensorView<float>& A,
                                 const choreo_ir::TensorView<float>& B,
                                 choreo_ir::TensorView<float>& C,
                                 const choreo_ir::compute::MatmulConfig& config,
                                 cudaStream_t stream);
}

using namespace choreo_ir;

/**
 * @brief Benchmark fixture for matrix multiplication
 */
class MatmulBenchmark : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        M = state.range(0);
        N = state.range(1);
        K = state.range(2);
        
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        
        // Create tensors
        A = Tensor<float>(Shape({M, K}), MemoryType::DEVICE);
        B = Tensor<float>(Shape({K, N}), MemoryType::DEVICE);
        C = Tensor<float>(Shape({M, N}), MemoryType::DEVICE);
        
        // Initialize with random data
        initialize_random_data();
        
        // Create CUDA events for timing
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    void TearDown(const benchmark::State& state) override {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
protected:
    void initialize_random_data() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Create host data
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        
        for (auto& val : h_A) val = dis(gen);
        for (auto& val : h_B) val = dis(gen);
        
        // Copy to device
        A.copy_from_host(h_A.data());
        B.copy_from_host(h_B.data());
        C.fill(0.0f);
    }
    
    double get_tflops(double time_ms) const {
        double flops = 2.0 * M * N * K; // Each element: K multiply-adds
        return (flops / 1e12) / (time_ms / 1000.0);
    }
    
    index_t M, N, K;
    Tensor<float> A, B, C;
    cudaEvent_t start_event, stop_event;
    cudaStream_t stream;
};

/**
 * @brief Benchmark our naive kernel implementation
 */
BENCHMARK_DEFINE_F(MatmulBenchmark, NaiveKernel)(benchmark::State& state) {
    auto config = compute::MatmulConfig{};
    config.algorithm = compute::MatmulAlgorithm::NAIVE;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    // Launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);
    
    for (auto _ : state) {
        // Reset output
        C.fill(0.0f);
        
        // Time the kernel
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        
        // Launch our naive kernel
        matmul_naive_kernel_float<<<grid_size, block_size, 0, stream>>>(
            A.data(), B.data(), C.data(),
            M, N, K,
            A.stride()[0], B.stride()[0], C.stride()[0],
            config.alpha, config.beta
        );
        
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        
        state.SetIterationTime(time_ms / 1000.0);
        state.counters["TFLOPS"] = get_tflops(time_ms);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}

/**
 * @brief Benchmark our shared memory kernel implementation
 */
BENCHMARK_DEFINE_F(MatmulBenchmark, SharedKernel)(benchmark::State& state) {
    auto config = compute::MatmulConfig{};
    config.algorithm = compute::MatmulAlgorithm::SHARED_MEMORY;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    config.tile_m = 16;
    config.tile_n = 16;
    config.tile_k = 16;
    
    // Launch configuration
    dim3 block_size(config.tile_n, config.tile_m);
    dim3 grid_size((N + config.tile_n - 1) / config.tile_n,
                   (M + config.tile_m - 1) / config.tile_m);
    
    for (auto _ : state) {
        // Reset output
        C.fill(0.0f);
        
        // Time the kernel
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        
        // Launch our shared memory kernel
        matmul_shared_kernel_float<<<grid_size, block_size, 0, stream>>>(
            A.data(), B.data(), C.data(),
            M, N, K,
            A.stride()[0], B.stride()[0], C.stride()[0],
            config.alpha, config.beta
        );
        
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        
        state.SetIterationTime(time_ms / 1000.0);
        state.counters["TFLOPS"] = get_tflops(time_ms);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}

/**
 * @brief Benchmark cuBLAS implementation
 */
BENCHMARK_DEFINE_F(MatmulBenchmark, CuBLAS)(benchmark::State& state) {
    auto config = compute::MatmulConfig{};
    config.algorithm = compute::MatmulAlgorithm::CUBLAS;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    for (auto _ : state) {
        // Reset output
        C.fill(0.0f);
        
        // Time the operation
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        
        // Call cuBLAS implementation
        cublas_matmul_impl_float(A.view(), B.view(), C.view(), config, stream);
        
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        
        state.SetIterationTime(time_ms / 1000.0);
        state.counters["TFLOPS"] = get_tflops(time_ms);
        state.counters["M"] = M;
        state.counters["N"] = N;
        state.counters["K"] = K;
    }
}

// Register benchmarks with different matrix sizes
BENCHMARK_REGISTER_F(MatmulBenchmark, NaiveKernel)
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MatmulBenchmark, SharedKernel)
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MatmulBenchmark, CuBLAS)
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

// Rectangular matrices
BENCHMARK_REGISTER_F(MatmulBenchmark, NaiveKernel)
    ->Args({1024, 2048, 512})
    ->Args({2048, 1024, 512})
    ->Args({512, 4096, 1024})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MatmulBenchmark, SharedKernel)
    ->Args({1024, 2048, 512})
    ->Args({2048, 1024, 512})
    ->Args({512, 4096, 1024})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(MatmulBenchmark, CuBLAS)
    ->Args({1024, 2048, 512})
    ->Args({2048, 1024, 512})
    ->Args({512, 4096, 1024})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN(); 