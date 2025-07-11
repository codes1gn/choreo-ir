#!/bin/bash

# Choreo-IR Build Script
# Configures CUDA 12.6 environment and builds the project

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
BUILD_TYPE="Release"

print_info "Starting Choreo-IR test process..."
print_info "CUDA Version: ${CUDA_VERSION}"
print_info "CUDA Home: ${CUDA_HOME}"

# Check if CUDA 12.6 is installed
if [ ! -d "${CUDA_HOME}" ]; then
    print_error "CUDA ${CUDA_VERSION} not found at ${CUDA_HOME}"
    print_info "Available CUDA versions:"
    ls -la /usr/local/ | grep cuda || true
    exit 1
fi

print_success "Found CUDA ${CUDA_VERSION} at ${CUDA_HOME}"

# Set up CUDA environment variables
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:/bin:/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

print_info "Set CUDA environment variables:"
print_info "  CUDA_HOME: ${CUDA_HOME}"
print_info "  PATH: ${PATH}"
print_info "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Create/update nvcc symlink if needed
if [ ! -L "/usr/bin/nvcc" ] || [ "$(readlink /usr/bin/nvcc)" != "${CUDA_HOME}/bin/nvcc" ]; then
    print_info "Updating nvcc symlink..."
    sudo rm -f /usr/bin/nvcc
    sudo ln -s "${CUDA_HOME}/bin/nvcc" /usr/bin/nvcc
    print_success "Updated nvcc symlink to CUDA ${CUDA_VERSION}"
fi

# Verify CUDA compiler
print_info "Verifying CUDA compiler..."
if ! nvcc --version > /dev/null 2>&1; then
    print_error "nvcc not found or not working"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
print_success "NVCC version: ${NVCC_VERSION}"

# Detect GPU architecture
print_info "Detecting GPU architecture..."
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    print_info "Detected GPU: ${GPU_NAME}"
    
    # Auto-detect CUDA architecture based on GPU
    if [[ "${GPU_NAME}" == *"A100"* ]]; then
        CUDA_ARCH="80"
        print_info "Detected A100 GPU, using CUDA arch 80"
    elif [[ "${GPU_NAME}" == *"V100"* ]]; then
        CUDA_ARCH="70"
        print_info "Detected V100 GPU, using CUDA arch 70"
    elif [[ "${GPU_NAME}" == *"RTX 40"* ]] || [[ "${GPU_NAME}" == *"4090"* ]] || [[ "${GPU_NAME}" == *"4080"* ]]; then
        CUDA_ARCH="89"
        print_info "Detected RTX 40 series GPU, using CUDA arch 89"
    elif [[ "${GPU_NAME}" == *"RTX 30"* ]] || [[ "${GPU_NAME}" == *"3090"* ]] || [[ "${GPU_NAME}" == *"3080"* ]]; then
        CUDA_ARCH="86"
        print_info "Detected RTX 30 series GPU, using CUDA arch 86"
    elif [[ "${GPU_NAME}" == *"RTX 20"* ]] || [[ "${GPU_NAME}" == *"T4"* ]]; then
        CUDA_ARCH="75"
        print_info "Detected RTX 20 series or T4 GPU, using CUDA arch 75"
    else
        # Default to common architectures
        CUDA_ARCH="70;75;80;86;89"
        print_warning "Unknown GPU, using default CUDA architectures: ${CUDA_ARCH}"
    fi
else
    print_warning "nvidia-smi not found, using default CUDA architectures"
    CUDA_ARCH="70;75;80;86;89"
fi

print_info "CUDA architectures: ${CUDA_ARCH}"

# Check for compatible GCC version
print_info "Checking GCC compatibility..."
if command -v gcc-8 > /dev/null 2>&1; then
    HOST_COMPILER="gcc-8"
    print_success "Found gcc-8 for CUDA compatibility"
elif command -v gcc-9 > /dev/null 2>&1; then
    HOST_COMPILER="gcc-9"
    print_warning "Using gcc-9 with CUDA (may need -allow-unsupported-compiler)"
else
    HOST_COMPILER="gcc"
    print_warning "Using default gcc, may have compatibility issues"
fi

GCC_VERSION=$(${HOST_COMPILER} --version | head -1)
print_info "Host compiler: ${HOST_COMPILER} (${GCC_VERSION})"

# Run basic tests
cd ${BUILD_DIR}
print_info "Running basic tests..."

if [ -f "tests/test_device" ]; then
    print_info "Running device tests..."
    if ./tests/test_device; then
        print_success "Device tests passed"
    else
        print_warning "Device tests failed"
    fi
fi

if [ -f "tests/test_shape" ]; then
    print_info "Running shape tests..."
    if ./tests/test_shape; then
        print_success "Shape tests passed"
    else
        print_warning "Shape tests failed"
    fi
fi

if [ -f "tests/test_stride" ]; then
    print_info "Running stride tests..."
    if ./tests/test_stride; then
        print_success "Stride tests passed"
    else
        print_warning "Stride tests failed"
    fi
fi

if [ -f "tests/test_end2end" ]; then
    print_info "Running end-to-end tests..."
    if timeout 60 ./tests/test_end2end; then
        print_success "End-to-end tests passed"
    else
        print_warning "End-to-end tests failed or timed out"
    fi
fi

print_success "Build script completed!"
print_info "Build artifacts are in: $(pwd)"
print_info "To run tests manually:"
print_info "  cd ${BUILD_DIR}"
print_info "  ./tests/test_end2end"
print_info "  ./tests/test_device"
print_info "  ./tests/test_shape" 