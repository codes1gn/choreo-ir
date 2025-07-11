#!/bin/bash

# Simple script to run test_end2end with proper CUDA environment

set -e  # Exit on any error

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

print_info "Running test_end2end with CUDA ${CUDA_VERSION}..."

# Set up CUDA environment variables
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:/bin:/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    print_error "Build directory '${BUILD_DIR}' not found. Please run ./scripts/build.sh first."
    exit 1
fi

# Check if test_end2end exists
if [ ! -f "${BUILD_DIR}/tests/test_end2end" ]; then
    print_error "test_end2end not found. Please run ./scripts/build.sh first."
    exit 1
fi

# Show GPU info
if command -v nvidia-smi > /dev/null 2>&1; then
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    echo
fi

# Run the test
print_info "Executing test_end2end..."
cd "${BUILD_DIR}"

# Run with timeout to prevent hanging
if timeout 120 ./tests/test_end2end; then
    print_success "test_end2end completed successfully!"
    exit 0
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        print_error "test_end2end timed out after 120 seconds"
    else
        print_error "test_end2end failed with exit code: $EXIT_CODE"
    fi
    exit $EXIT_CODE
fi 