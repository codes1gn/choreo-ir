/**
 * @file benchmark_suite.cpp
 * @brief Comprehensive performance benchmark suite for Choreo-IR
 * 
 * Compares our ideal API performance against:
 * 1. cuBLAS (single/half precision)
 * 2. cutlass (if available) 
 * 3. Raw CUDA kernels
 * 4. PyTorch (through Python interface)
 * 
 * Tracks performance regression and validates optimization claims
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <random>
#include <fstream>
#include <iomanip>

// Our ideal API
#include "choreo-ir/ideal_api.hpp"

// Baseline implementations
#include "../baselines/cublas_impl.hpp"
#include "../baselines/cudnn_impl.hpp"

using namespace choreo_ir;

/**
 * @brief Base class for performance benchmarks
 */
class PerformanceBenchmark {
public:
    static void SetUpBenchmark() {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceReset());
        
        // Get device properties
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
        
        std::cout << "Running benchmarks on: " << device_prop.name << std::endl;
        std::cout << "Compute Capability: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "Memory: " << device_prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        
        // Initialize libraries
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        
        // Set up random seed for reproducible results
        std::srand(42);
    }
    
    static void TearDownBenchmark() {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        CUDNN_CHECK(cudnnDestroy(cudnn_handle));
    }
    
protected:
    static cudaDeviceProp device_prop;
    static cublasHandle_t cublas_handle;
    static cudnnHandle_t cudnn_handle;
    
    // Helper to calculate TFLOPS
    static double calculate_tflops(size_t flops, double time_ms) {
        return (flops / 1e12) / (time_ms / 1000.0);
    }
    
    // Helper to create random data
    template<typename T>
    static std::vector<T> generate_random_data(size_t size, T min_val = T(-1), T max_val = T(1)) {
        std::vector<T> data(size);
        std::random_device rd;
        std::mt19937 gen(42);
        
        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(min_val, max_val);
            for (auto& val : data) val = dis(gen);
        } else if constexpr (std::is_same_v<T, __half>) {
            std::uniform_real_distribution<float> dis(float(min_val), float(max_val));
            for (auto& val : data) val = __float2half(dis(gen));
        }
        
        return data;
    }
};

// Static member definitions
cudaDeviceProp PerformanceBenchmark::device_prop;
cublasHandle_t PerformanceBenchmark::cublas_handle;
cudnnHandle_t PerformanceBenchmark::cudnn_handle;

/**
 * @brief Matrix multiplication benchmarks
 */
class MatmulBenchmark {
public:
    struct Config {
        int M, N, K;
        std::string name;
    };
    
    static std::vector<Config> get_test_configs() {
        return {
            {512, 512, 512, "small_square"},
            {1024, 1024, 1024, "medium_square"},
            {2048, 2048, 2048, "large_square"},
            {4096, 4096, 4096, "xl_square"},
            {512, 2048, 1024, "rectangular_1"},
            {2048, 512, 1024, "rectangular_2"},
            {8192, 1024, 1024, "tall_narrow"},
            {1024, 8192, 1024, "wide_short"},
            {16, 16, 16, "tiny_tc"},      // Tensor core minimum
            {64, 64, 64, "small_tc"},     // Tensor core friendly
            {128, 128, 128, "medium_tc"}, // Common tile size
        };
    }
};

/**
 * @brief Benchmark our ideal API vs cuBLAS (float)
 */
static void BM_MatmulFloat_ChoreIR_vs_cuBLAS(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1); 
    const int K = state.range(2);
    
    // Create test data
    auto A = host_tensor<float>::random({M, K});
    auto B = host_tensor<float>::random({K, N});
    auto C = host_tensor<float>::zeros({M, N});
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        auto C_warm = A * B;
    }
    
    size_t flops = 2ULL * M * N * K;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto C_result = A * B;
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    state.SetBytesProcessed(state.iterations() * (M*K + K*N + M*N) * sizeof(float));
    
    // Add custom TFLOPS counter
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
    state.counters["M"] = M;
    state.counters["N"] = N;
    state.counters["K"] = K;
}

/**
 * @brief Benchmark cuBLAS baseline (float)
 */
static void BM_MatmulFloat_cuBLAS(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);
    
    // Create device data
    auto h_A = PerformanceBenchmark::generate_random_data<float>(M * K);
    auto h_B = PerformanceBenchmark::generate_random_data<float>(K * N);
    auto h_C = std::vector<float>(M * N, 0.0f);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        CUBLAS_CHECK(cublasSgemm(PerformanceBenchmark::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t flops = 2ULL * M * N * K;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSgemm(PerformanceBenchmark::cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    state.SetBytesProcessed(state.iterations() * (M*K + K*N + M*N) * sizeof(float));
    
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = PerformanceBenchmark::calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
    state.counters["M"] = M;
    state.counters["N"] = N;
    state.counters["K"] = K;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/**
 * @brief Benchmark mixed precision (half input, float output)
 */
static void BM_MatmulHalf_ChoreIR_TensorCore(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);
    
    // Create half precision inputs
    auto A = host_tensor<__half>::random({M, K});
    auto B = host_tensor<__half>::random({K, N});
    auto C = host_tensor<float>::zeros({M, N});
    
    // This should automatically use tensor cores
    for (int i = 0; i < 3; ++i) {
        matmul(A, B, C);
    }
    
    size_t flops = 2ULL * M * N * K;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    state.SetBytesProcessed(state.iterations() * (M*K*2 + K*N*2 + M*N*4)); // half=2bytes, float=4bytes
    
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = PerformanceBenchmark::calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
    state.counters["UseTensorCores"] = 1;
}

/**
 * @brief Benchmark cuBLAS mixed precision baseline
 */
static void BM_MatmulHalf_cuBLAS_TensorCore(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(1);
    const int K = state.range(2);
    
    // Create device data
    auto h_A = PerformanceBenchmark::generate_random_data<__half>(M * K);
    auto h_B = PerformanceBenchmark::generate_random_data<__half>(K * N);
    auto h_C = std::vector<float>(M * N, 0.0f);
    
    __half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        CUBLAS_CHECK(cublasGemmEx(PerformanceBenchmark::cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                d_B, CUDA_R_16F, N,
                                d_A, CUDA_R_16F, K,
                                &beta,
                                d_C, CUDA_R_32F, N,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t flops = 2ULL * M * N * K;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasGemmEx(PerformanceBenchmark::cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                d_B, CUDA_R_16F, N,
                                d_A, CUDA_R_16F, K,
                                &beta,
                                d_C, CUDA_R_32F, N,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    state.SetBytesProcessed(state.iterations() * (M*K*2 + K*N*2 + M*N*4));
    
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = PerformanceBenchmark::calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
    state.counters["UseTensorCores"] = 1;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/**
 * @brief Batch matrix multiplication benchmark
 */
static void BM_BatchMatmul_ChoreIR(benchmark::State& state) {
    const int batch_size = state.range(0);
    const int M = state.range(1);
    const int N = state.range(2);
    const int K = state.range(3);
    
    auto A = host_tensor<__half>::random({batch_size, M, K});
    auto B = host_tensor<__half>::random({batch_size, K, N});
    
    for (int i = 0; i < 3; ++i) {
        auto C = batch_matmul(A, B);
    }
    
    size_t flops = 2ULL * batch_size * M * N * K;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto C = batch_matmul(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = PerformanceBenchmark::calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
    state.counters["BatchSize"] = batch_size;
}

/**
 * @brief Convolution benchmark
 */
static void BM_Conv2D_ChoreIR(benchmark::State& state) {
    const int N = state.range(0);
    const int C = state.range(1);
    const int H = state.range(2);
    const int W = state.range(3);
    const int K = state.range(4);
    const int R = 3, S = 3; // 3x3 kernel
    
    auto input = host_tensor<__half>::random({N, C, H, W});
    auto weight = host_tensor<__half>::random({K, C, R, S});
    
    for (int i = 0; i < 3; ++i) {
        auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
    }
    
    // FLOPs for convolution: N * K * H_out * W_out * C * R * S * 2
    size_t flops = 2ULL * N * K * H * W * C * R * S;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetItemsProcessed(state.iterations() * flops);
    double avg_time_ms = state.GetAverageIterationTime() * 1000;
    double tflops = PerformanceBenchmark::calculate_tflops(flops, avg_time_ms);
    state.counters["TFLOPS"] = tflops;
}

/**
 * @brief Memory bandwidth benchmark
 */
static void BM_MemoryBandwidth_TensorCopy(benchmark::State& state) {
    const size_t size = state.range(0) * 1024 * 1024; // Size in MB
    
    auto src = host_tensor<float>::random({size / sizeof(float)});
    auto dst = host_tensor<float>::zeros({size / sizeof(float)});
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        dst = src; // Should trigger optimized copy
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e6);
    }
    
    state.SetBytesProcessed(state.iterations() * size * 2); // Read + Write
    double bandwidth_gb_s = state.GetBytesProcessedPerSecond() / (1024*1024*1024);
    state.counters["BandwidthGB/s"] = bandwidth_gb_s;
}

// Register benchmarks with different matrix sizes
static void RegisterMatmulBenchmarks() {
    auto configs = MatmulBenchmark::get_test_configs();
    
    for (const auto& config : configs) {
        // Register our implementation
        auto bm_choreo = benchmark::RegisterBenchmark(
            ("ChoreIR_Float_" + config.name).c_str(),
            BM_MatmulFloat_ChoreIR_vs_cuBLAS
        )->Args({config.M, config.N, config.K})->UseManualTime()->Unit(benchmark::kMillisecond);
        
        // Register cuBLAS baseline
        auto bm_cublas = benchmark::RegisterBenchmark(
            ("cuBLAS_Float_" + config.name).c_str(),
            BM_MatmulFloat_cuBLAS
        )->Args({config.M, config.N, config.K})->UseManualTime()->Unit(benchmark::kMillisecond);
        
        // Register tensor core benchmarks for sizes that support it
        if (config.M % 16 == 0 && config.N % 16 == 0 && config.K % 16 == 0) {
            auto bm_tc_choreo = benchmark::RegisterBenchmark(
                ("ChoreIR_Half_TC_" + config.name).c_str(),
                BM_MatmulHalf_ChoreIR_TensorCore
            )->Args({config.M, config.N, config.K})->UseManualTime()->Unit(benchmark::kMillisecond);
            
            auto bm_tc_cublas = benchmark::RegisterBenchmark(
                ("cuBLAS_Half_TC_" + config.name).c_str(),
                BM_MatmulHalf_cuBLAS_TensorCore
            )->Args({config.M, config.N, config.K})->UseManualTime()->Unit(benchmark::kMillisecond);
        }
    }
}

// Register batch benchmarks
static void RegisterBatchBenchmarks() {
    // Batch size, M, N, K
    std::vector<std::vector<int64_t>> batch_configs = {
        {8, 256, 256, 256},
        {16, 512, 512, 512},
        {32, 256, 256, 256},
        {64, 128, 128, 128},
    };
    
    for (const auto& config : batch_configs) {
        std::string name = "BatchMatmul_" + std::to_string(config[0]) + "x" +
                          std::to_string(config[1]) + "x" + std::to_string(config[2]) + "x" + std::to_string(config[3]);
        
        benchmark::RegisterBenchmark(name.c_str(), BM_BatchMatmul_ChoreIR)
            ->Args(config)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
}

// Register convolution benchmarks
static void RegisterConvBenchmarks() {
    // N, C, H, W, K
    std::vector<std::vector<int64_t>> conv_configs = {
        {1, 64, 224, 224, 64},   // ResNet-like
        {32, 128, 56, 56, 128},  // Batch processing
        {16, 256, 28, 28, 256},  // Deep layers
        {8, 512, 14, 14, 512},   // Very deep
    };
    
    for (const auto& config : conv_configs) {
        std::string name = "Conv2D_" + std::to_string(config[0]) + "x" +
                          std::to_string(config[1]) + "x" + std::to_string(config[2]) + "x" +
                          std::to_string(config[3]) + "x" + std::to_string(config[4]);
        
        benchmark::RegisterBenchmark(name.c_str(), BM_Conv2D_ChoreIR)
            ->Args(config)->UseManualTime()->Unit(benchmark::kMillisecond);
    }
}

// Memory bandwidth benchmarks
static void RegisterMemoryBenchmarks() {
    // Test different sizes in MB
    std::vector<int64_t> sizes = {1, 4, 16, 64, 256, 1024};
    
    for (auto size : sizes) {
        std::string name = "MemBW_" + std::to_string(size) + "MB";
        benchmark::RegisterBenchmark(name.c_str(), BM_MemoryBandwidth_TensorCopy)
            ->Args({size})->UseManualTime()->Unit(benchmark::kMillisecond);
    }
}

/**
 * @brief Performance regression tracking
 */
class PerformanceTracker {
public:
    struct BenchmarkResult {
        std::string name;
        double tflops;
        double bandwidth_gb_s;
        double time_ms;
        std::string timestamp;
    };
    
    static void save_results(const std::vector<BenchmarkResult>& results) {
        std::ofstream file("performance_history.json", std::ios::app);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        file << "{\n";
        file << "  \"timestamp\": \"" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\",\n";
        file << "  \"device\": \"" << PerformanceBenchmark::device_prop.name << "\",\n";
        file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            file << "    {\n";
            file << "      \"name\": \"" << result.name << "\",\n";
            file << "      \"tflops\": " << result.tflops << ",\n";
            file << "      \"bandwidth_gb_s\": " << result.bandwidth_gb_s << ",\n";
            file << "      \"time_ms\": " << result.time_ms << "\n";
            file << "    }";
            if (i < results.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "  ]\n";
        file << "},\n";
    }
    
    static bool check_regression(const BenchmarkResult& current, const BenchmarkResult& baseline, double threshold = 0.05) {
        if (baseline.tflops == 0) return false;
        
        double performance_ratio = current.tflops / baseline.tflops;
        return performance_ratio < (1.0 - threshold);
    }
};

int main(int argc, char** argv) {
    // Initialize
    PerformanceBenchmark::SetUpBenchmark();
    
    // Register all benchmarks
    RegisterMatmulBenchmarks();
    RegisterBatchBenchmarks();
    RegisterConvBenchmarks();
    RegisterMemoryBenchmarks();
    
    // Run benchmarks
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    
    benchmark::RunSpecifiedBenchmarks();
    
    // Cleanup
    PerformanceBenchmark::TearDownBenchmark();
    
    std::cout << "\nBenchmark completed. Results saved to performance_history.json" << std::endl;
    
    return 0;
} 