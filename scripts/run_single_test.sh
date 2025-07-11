#!/bin/bash

# Script to run individual test cases to avoid GPU resource conflicts

set -e

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
CUDA_VERSION="12.6"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
BUILD_DIR="build"

# Available test cases
AVAILABLE_TESTS=(
    "ChoreoIRTest.TensorCreationAndProperties"
    "ChoreoIRTest.SimpleMatrixMultiplication"
    "ChoreoIRTest.MixedPrecisionMatmul"
    "ChoreoIRTest.BatchOperations"
    "ChoreoIRTest.ConvolutionOperations"
    "ChoreoIRTest.ShapeOperations"
    "ChoreoIRTest.LayoutOperations"
    "ChoreoIRTest.StrideCalculations"
    "ChoreoIRTest.ElementWiseOperations"
    "ChoreoIRTest.RealWorldUsagePattern"
    "ChoreoIRTest.PerformanceCharacteristics"
    "ChoreoIRTest.DifferentDataTypes"
    "ChoreoIRTest.MemoryManagement"
    "ChoreoIRTest.ErrorHandling"
)

# Function to setup CUDA environment
setup_cuda_env() {
    export CUDA_HOME="${CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:/bin:/usr/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=0
}

# Function to run a single test
run_single_test() {
    local test_name="$1"
    
    print_info "Setting up CUDA environment..."
    setup_cuda_env
    
    print_info "Cleaning GPU state before test..."
    pkill -f "test_end2end" 2>/dev/null || true
    sleep 1
    
    print_info "Running test: ${test_name}"
    print_info "Command: ./tests/test_end2end --gtest_filter=${test_name}"
    
    cd "${BUILD_DIR}"
    
    if ./tests/test_end2end --gtest_filter="${test_name}"; then
        print_success "Test ${test_name} PASSED"
        return 0
    else
        print_error "Test ${test_name} FAILED"
        return 1
    fi
}

# Function to run all tests sequentially
run_all_sequential() {
    local passed=0
    local failed=0
    local failed_tests=()
    
    print_info "Running all tests sequentially to avoid GPU conflicts..."
    
    for test in "${AVAILABLE_TESTS[@]}"; do
        print_info ""
        print_info "========================================="
        print_info "Running: $test"
        print_info "========================================="
        
        if run_single_test "$test"; then
            ((passed++))
        else
            ((failed++))
            failed_tests+=("$test")
        fi
        
        # Clean up between tests
        sleep 2
    done
    
    print_info ""
    print_info "========================================="
    print_info "FINAL RESULTS"
    print_info "========================================="
    print_success "PASSED: $passed tests"
    print_error "FAILED: $failed tests"
    
    if [ $failed -gt 0 ]; then
        print_error "Failed tests:"
        for failed_test in "${failed_tests[@]}"; do
            print_error "  - $failed_test"
        done
    fi
    
    return $failed
}

# Function to show available tests
show_tests() {
    print_info "Available test cases:"
    for i in "${!AVAILABLE_TESTS[@]}"; do
        printf "%2d: %s\n" $((i+1)) "${AVAILABLE_TESTS[$i]}"
    done
}

# Main script logic
if [ $# -eq 0 ]; then
    print_info "Usage: $0 [test_name|--all|--list]"
    print_info ""
    print_info "Examples:"
    print_info "  $0 --list                                    # Show available tests"
    print_info "  $0 --all                                     # Run all tests sequentially"
    print_info "  $0 ChoreoIRTest.SimpleMatrixMultiplication  # Run specific test"
    print_info ""
    show_tests
    exit 0
fi

case "$1" in
    --list)
        show_tests
        ;;
    --all)
        run_all_sequential
        ;;
    *)
        # Check if the test name is valid
        if [[ " ${AVAILABLE_TESTS[@]} " =~ " $1 " ]]; then
            run_single_test "$1"
        else
            print_error "Unknown test: $1"
            print_info "Use --list to see available tests"
            exit 1
        fi
        ;;
esac 