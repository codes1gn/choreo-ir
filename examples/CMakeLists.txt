# Examples directory CMakeLists.txt

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Function to create example executable
function(add_example example_name source_file)
    add_executable(${example_name} ${source_file})
    
    target_link_libraries(${example_name}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
    )
    
    # Set CUDA properties
    set_property(TARGET ${example_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${example_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Set output directory
    set_target_properties(${example_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Ideal API examples
add_example(simple_ideal_example simple_ideal_example.cu)
add_example(ideal_matmul_example ideal_matmul_example.cu)

# Add custom target to build all examples
add_custom_target(examples)
add_dependencies(examples simple_ideal_example ideal_matmul_example) 