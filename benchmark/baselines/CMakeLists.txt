# Baseline implementations for comparison

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Create baseline library
add_library(choreo-ir-baselines
    cublas_impl.hpp
    cudnn_impl.hpp
)

# Make it header-only
set_target_properties(choreo-ir-baselines PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(choreo-ir-baselines
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(choreo-ir-baselines
    PUBLIC
    CUDA::cublas
    CUDA::cudart
)

target_compile_features(choreo-ir-baselines PUBLIC cxx_std_14)

# Add compile definitions
target_compile_definitions(choreo-ir-baselines PUBLIC
    HAVE_CUBLAS
) 