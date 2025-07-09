#!/usr/bin/env python3
"""
Quick demonstration of Choreo-IR ideal programming experience

This script:
1. Builds the project
2. Runs a subset of tests to verify functionality
3. Runs performance benchmarks on key operations
4. Shows the ideal API in action
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run a command and return the result"""
    print(f"🔄 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=True)
    if capture_output:
        return result.stdout, result.stderr
    return result

def build_project(build_dir="build", clean=False):
    """Build the Choreo-IR project"""
    print("🔨 Building Choreo-IR...")
    
    build_path = Path(build_dir)
    
    if clean and build_path.exists():
        import shutil
        shutil.rmtree(build_path)
    
    build_path.mkdir(exist_ok=True)
    
    # Configure
    run_command([
        "cmake", "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=75;80;86",  # Support modern GPUs
        "-DENABLE_TESTING=ON",
        "-DENABLE_BENCHMARKS=ON"
    ], cwd=build_path)
    
    # Build
    run_command(["cmake", "--build", ".", "--parallel", "4"], cwd=build_path)
    
    print("✅ Build completed successfully!")

def run_tests(build_dir="build"):
    """Run the test suite"""
    print("🧪 Running tests...")
    
    try:
        # Run unit tests
        run_command(["ctest", "--output-on-failure", "-L", "unit"], cwd=build_dir)
        print("✅ Unit tests passed!")
        
        # Run a quick smoke test for benchmarks
        run_command(["ctest", "--output-on-failure", "-L", "smoke"], cwd=build_dir)
        print("✅ Benchmark smoke test passed!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with return code {e.returncode}")
        return False
    
    return True

def run_quick_benchmarks(build_dir="build"):
    """Run a quick set of benchmarks to show performance"""
    print("🚀 Running quick performance benchmarks...")
    
    benchmark_exe = Path(build_dir) / "benchmark" / "performance" / "benchmark_suite"
    
    if not benchmark_exe.exists():
        print(f"❌ Benchmark executable not found: {benchmark_exe}")
        return False
    
    try:
        # Run a subset of benchmarks for demonstration
        stdout, stderr = run_command([
            str(benchmark_exe),
            "--benchmark_filter=.*small.*|.*medium.*",
            "--benchmark_format=console",
            "--benchmark_repetitions=3",
            "--benchmark_report_aggregates_only=true"
        ], cwd=build_dir, capture_output=True)
        
        print("📊 Benchmark Results:")
        print(stdout)
        
        if stderr:
            print("⚠️ Benchmark warnings:")
            print(stderr)
        
        print("✅ Quick benchmarks completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmarks failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def show_api_demo():
    """Show example code demonstrating the ideal API"""
    print("🎯 Choreo-IR Ideal Programming Experience Demo")
    print("=" * 50)
    
    demo_code = '''
// Ideal Choreo-IR usage - as simple as mathematical notation!

#include "choreo-ir/ideal_api.hpp"
using namespace choreo_ir;

int main() {
    // Create matrices with intuitive syntax
    auto A = host_tensor<float>::random({1024, 512});
    auto B = host_tensor<float>::random({512, 256});
    
    // Matrix multiplication - just like math!
    auto C = A * B;
    
    // Mixed precision with automatic tensor core usage
    auto A_half = host_tensor<__half>::random({2048, 1024});
    auto B_half = host_tensor<__half>::random({1024, 512});
    auto C_float = host_tensor<float>::zeros({2048, 512});
    
    matmul(A_half, B_half, C_float);  // Automatically uses tensor cores!
    
    // Batch operations are natural
    auto batch_A = host_tensor<__half>::random({32, 256, 256});
    auto batch_B = host_tensor<__half>::random({32, 256, 256});
    auto batch_C = batch_matmul(batch_A, batch_B);
    
    // Convolution is simple
    auto input = host_tensor<__half>::random({8, 64, 224, 224});
    auto weight = host_tensor<__half>::random({128, 64, 3, 3});
    auto output = conv2d(input, weight, /*stride=*/1, /*padding=*/1);
    
    // Chain operations naturally
    auto result = (A * B) + bias;  // Broadcasting works automatically
    
    return 0;
}
'''
    
    print("📝 Example Code:")
    print(demo_code)
    
    print("\n🎯 Key Features Demonstrated:")
    print("✓ Natural mathematical syntax: C = A * B")
    print("✓ Automatic tensor core usage for half precision")
    print("✓ Intuitive batch operations")
    print("✓ Simple convolution API")
    print("✓ Automatic memory management")
    print("✓ Broadcasting support")
    print("✓ Mixed precision support")

def check_cuda_environment():
    """Check if CUDA environment is available"""
    print("🔍 Checking CUDA environment...")
    
    try:
        # Check nvidia-smi
        stdout, _ = run_command(["nvidia-smi"], capture_output=True)
        print("✅ NVIDIA GPU detected")
        
        # Extract GPU info (simplified)
        lines = stdout.split('\n')
        for line in lines:
            if 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                print(f"🖥️  GPU: {line.strip()}")
                break
        
        # Check nvcc
        stdout, _ = run_command(["nvcc", "--version"], capture_output=True)
        version_line = [line for line in stdout.split('\n') if 'release' in line]
        if version_line:
            print(f"🔧 CUDA: {version_line[0].strip()}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ CUDA not available or not properly installed")
        print("💡 This demo requires CUDA for GPU operations")
        return False

def main():
    parser = argparse.ArgumentParser(description="Choreo-IR Quick Demo")
    parser.add_argument("--build-dir", default="build", help="Build directory")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark execution")
    parser.add_argument("--cuda-check", action="store_true", help="Only check CUDA environment")
    
    args = parser.parse_args()
    
    print("🎯 Choreo-IR Ideal Programming Experience Demo")
    print("=" * 60)
    
    if args.cuda_check:
        return 0 if check_cuda_environment() else 1
    
    # Check CUDA environment
    if not check_cuda_environment():
        print("⚠️  Continuing without CUDA (limited functionality)")
    
    # Show the ideal API demo
    show_api_demo()
    
    success = True
    
    # Build project
    if not args.skip_build:
        try:
            build_project(args.build_dir, args.clean)
        except subprocess.CalledProcessError:
            print("❌ Build failed")
            return 1
    
    # Run tests
    if not args.skip_tests:
        if not run_tests(args.build_dir):
            success = False
    
    # Run benchmarks
    if not args.skip_benchmarks:
        if not run_quick_benchmarks(args.build_dir):
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Demo completed successfully!")
        print("🚀 Your ideal CUDA programming experience is ready!")
        print(f"📂 Build artifacts in: {args.build_dir}/")
        print("📖 Next steps:")
        print("   - Explore examples/ directory for more usage patterns")
        print("   - Run full benchmark suite with 'make run_all_performance_benchmarks'")
        print("   - Check performance regression with 'python scripts/run_regression_tests.py'")
    else:
        print("❌ Demo encountered some issues")
        print("🔍 Check the error messages above for troubleshooting")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 