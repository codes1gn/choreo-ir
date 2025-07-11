#!/bin/bash

# GPU Status Check Script
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

print_info "Checking GPU status..."

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found!"
    exit 1
fi

print_info "GPU Information:"
nvidia-smi

print_info ""
print_info "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

print_info ""
print_info "Running processes on GPU:"
nvidia-smi pmon -i 0 -c 1

print_info ""
print_info "Checking for zombie CUDA processes..."
ps aux | grep -v grep | grep -E "(cuda|nvcc|test_)"

print_info ""
print_info "Attempting to reset GPU state..."
# Try to kill any hanging CUDA processes
pkill -f "test_end2end" 2>/dev/null || true
pkill -f "cuda" 2>/dev/null || true

# Wait a moment
sleep 2

print_info ""
print_info "GPU status after cleanup:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

print_success "GPU check completed" 