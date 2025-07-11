# Tests for Choreo-IR ideal programming experience

# Find required packages
find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# Create core library target (header-only interface library)
add_library(choreo-ir-core INTERFACE)
target_include_directories(choreo-ir-core 
    INTERFACE 
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(choreo-ir-core 
    INTERFACE
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
# Set C++17 for the core library
target_compile_features(choreo-ir-core INTERFACE cxx_std_17)

# Create device management test
add_executable(test_device
    test_device.cpp
)

target_link_libraries(test_device
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
    ${CUDA_LIBRARIES}
)

target_include_directories(test_device
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Set C++17 for device test
set_property(TARGET test_device PROPERTY CXX_STANDARD 17)
set_property(TARGET test_device PROPERTY CXX_STANDARD_REQUIRED ON)

# Create debug test for matrix multiplication
add_executable(test_debug_simple
    test_debug_simple.cpp
)

target_link_libraries(test_debug_simple
    PRIVATE
    choreo-ir-core
    ${CUDA_LIBRARIES}
)

target_include_directories(test_debug_simple
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Set C++17 for debug test
set_property(TARGET test_debug_simple PROPERTY CXX_STANDARD 17)
set_property(TARGET test_debug_simple PROPERTY CXX_STANDARD_REQUIRED ON)

# Add CUDA support if available
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # 注释掉 CUDA_STANDARD 17，避免语言标准要求
    # set_property(TARGET test_device PROPERTY CUDA_STANDARD 17)
    # set_property(TARGET test_device PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

# Add compiler-specific flags
target_compile_options(test_device PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Register device test with CTest
add_test(NAME DeviceTest COMMAND test_device)
set_tests_properties(DeviceTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;device;cuda"
)

# Create test for end2end API
add_executable(test_end2end
    test_end2end.cpp
)

target_link_libraries(test_end2end
    PRIVATE 
        choreo-ir-core
        choreo-ir-baselines
        GTest::gtest
        GTest::gtest_main
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        Threads::Threads
)

target_include_directories(test_end2end
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS}
)

# Set C++17 standard for test_end2end
set_property(TARGET test_end2end PROPERTY CXX_STANDARD 17)
set_property(TARGET test_end2end PROPERTY CXX_STANDARD_REQUIRED ON)

# CUDA configuration for test_end2end
if(CMAKE_CUDA_COMPILER)
    # 注释掉 CUDA_STANDARD 17，避免语言标准要求
    # set_property(TARGET test_end2end PROPERTY CUDA_STANDARD 17)
    # set_property(TARGET test_end2end PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

# Add compiler options for test_end2end
target_compile_options(test_end2end PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Register the test
add_test(NAME End2EndAPITest COMMAND test_end2end)

# Set test properties
set_tests_properties(End2EndAPITest PROPERTIES
    TIMEOUT 300
    LABELS "unit;api;cuda"
)

# Add custom test for specific GPU architectures
if(CMAKE_CUDA_COMPILER)
    # Test for tensor core support (requires compute capability >= 7.0)
    add_test(NAME End2EndAPITest_TensorCore 
        COMMAND test_end2end --gtest_filter="*TensorCore*"
    )
    set_tests_properties(End2EndAPITest_TensorCore PROPERTIES
        TIMEOUT 600
        LABELS "unit;api;cuda;tensor_core"
    )
    
    # Test for large matrices (requires sufficient GPU memory)
    add_test(NAME End2EndAPITest_LargeMatrix
        COMMAND test_end2end --gtest_filter="*LargeMatrix*"
    )
    set_tests_properties(End2EndAPITest_LargeMatrix PROPERTIES
        TIMEOUT 900
        LABELS "unit;api;cuda;large_memory"
    )
endif()

# Create shape test executable
add_executable(test_shape
    test_shape.cpp
)

target_link_libraries(test_shape
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_shape
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_shape PROPERTY CXX_STANDARD 17)
set_property(TARGET test_shape PROPERTY CXX_STANDARD_REQUIRED ON)

# Add compiler-specific flags
target_compile_options(test_shape PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Register shape test with CTest
add_test(NAME ShapeTest COMMAND test_shape)
set_tests_properties(ShapeTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;tensor;shape"
)

# Create stride test executable
add_executable(test_stride
    test_stride.cpp
)

target_link_libraries(test_stride
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_stride
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_stride PROPERTY CXX_STANDARD 17)
set_property(TARGET test_stride PROPERTY CXX_STANDARD_REQUIRED ON)

# Add compiler-specific flags
target_compile_options(test_stride PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Register stride test with CTest
add_test(NAME StrideTest COMMAND test_stride)
set_tests_properties(StrideTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;tensor;stride"
)

# Create layout test executable
add_executable(test_layout
    test_layout.cpp
)

target_link_libraries(test_layout
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_layout
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_layout PROPERTY CXX_STANDARD 17)
set_property(TARGET test_layout PROPERTY CXX_STANDARD_REQUIRED ON)

# Add compiler-specific flags
target_compile_options(test_layout PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Register layout test with CTest
add_test(NAME LayoutTest COMMAND test_layout)
set_tests_properties(LayoutTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;tensor;layout"
)

# Create tensor test executable
add_executable(test_tensor
    test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_tensor
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_tensor PROPERTY CXX_STANDARD 17)
set_property(TARGET test_tensor PROPERTY CXX_STANDARD_REQUIRED ON)

# Add compiler-specific flags
target_compile_options(test_tensor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Register tensor test with CTest
add_test(NAME TensorTest COMMAND test_tensor)
set_tests_properties(TensorTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;tensor;core"
)

# Create cuda_utils test executable
add_executable(test_cuda_utils
    test_cuda_utils.cpp
)

target_link_libraries(test_cuda_utils
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
    ${CUDA_LIBRARIES}
)

target_include_directories(test_cuda_utils
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
)

set_property(TARGET test_cuda_utils PROPERTY CXX_STANDARD 17)
set_property(TARGET test_cuda_utils PROPERTY CXX_STANDARD_REQUIRED ON)

# Add compiler-specific flags
target_compile_options(test_cuda_utils PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# Register cuda_utils test with CTest
add_test(NAME CudaUtilsTest COMMAND test_cuda_utils)
set_tests_properties(CudaUtilsTest PROPERTIES
    TIMEOUT 60
    LABELS "unit;cuda;utils"
)



# Create integration test executable
add_executable(test_integration
    test_integration.cpp
)

target_link_libraries(test_integration
    PRIVATE
    choreo-ir-core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_integration
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(TARGET test_integration PROPERTY CXX_STANDARD 17)
set_property(TARGET test_integration PROPERTY CXX_STANDARD_REQUIRED ON)

# Register integration test
add_test(NAME IntegrationTest COMMAND test_integration)
set_tests_properties(IntegrationTest PROPERTIES
    TIMEOUT 300
    LABELS "integration;system"
)

# Create custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_device test_shape test_stride test_layout test_tensor test_end2end test_integration test_cuda_core
    COMMENT "Running all tests"
)

# Create custom target for device tests only
add_custom_target(run_device_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L device
    DEPENDS test_device
    COMMENT "Running device management tests"
)

# Create custom target for tensor tests only
add_custom_target(run_tensor_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L tensor
    DEPENDS test_shape test_stride test_layout test_tensor
    COMMENT "Running tensor abstraction tests"
)

# Memory leak detection test (if available)
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_test(NAME End2EndAPITest_MemoryCheck
        COMMAND ${VALGRIND_EXECUTABLE} 
        --tool=memcheck 
        --leak-check=full 
        --error-exitcode=1
        $<TARGET_FILE:test_end2end>
    )
    set_tests_properties(End2EndAPITest_MemoryCheck PROPERTIES
        TIMEOUT 1800
        LABELS "unit;api;memory"
    )
endif()

# Performance comparison test
add_executable(test_performance_comparison
    test_performance_comparison.cpp
)

target_link_libraries(test_performance_comparison
    PRIVATE
    choreo-ir-core
    choreo-ir-baselines
    GTest::gtest
    GTest::gtest_main
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES}
)

target_include_directories(test_performance_comparison
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/baselines/include
)

set_property(TARGET test_performance_comparison PROPERTY CXX_STANDARD 17)
set_property(TARGET test_performance_comparison PROPERTY CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER)
    # 注释掉 CUDA_STANDARD 17，避免语言标准要求
    # set_property(TARGET test_performance_comparison PROPERTY CUDA_STANDARD 17)
    # set_property(TARGET test_performance_comparison PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

add_test(NAME PerformanceComparisonTest COMMAND test_performance_comparison)
set_tests_properties(PerformanceComparisonTest PROPERTIES
    TIMEOUT 1200
    LABELS "performance;comparison;cuda"
)

# Test coverage (if gcov is available)
if(CMAKE_COMPILER_IS_GNUCXX)
    option(ENABLE_COVERAGE "Enable test coverage" OFF)
    if(ENABLE_COVERAGE)
        target_compile_options(test_end2end PRIVATE --coverage)
        target_link_libraries(test_end2end PRIVATE --coverage)
        
        target_compile_options(test_integration PRIVATE --coverage)
        target_link_libraries(test_integration PRIVATE --coverage)
        
        # Add custom target for generating coverage report
        find_program(GCOV_EXECUTABLE gcov)
        find_program(LCOV_EXECUTABLE lcov)
        find_program(GENHTML_EXECUTABLE genhtml)
        
        if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
            add_custom_target(coverage
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '*/tests/*' --output-file coverage.info
                COMMAND ${GENHTML_EXECUTABLE} coverage.info --output-directory coverage_html
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating test coverage report"
            )
        endif()
    endif()
endif()

# Add sanitizer support for debugging
option(ENABLE_SANITIZERS "Enable sanitizers (AddressSanitizer, UBSan)" OFF)
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
        
        target_compile_options(test_end2end PRIVATE ${SANITIZER_FLAGS})
        target_link_options(test_end2end PRIVATE ${SANITIZER_FLAGS})
        
        target_compile_options(test_integration PRIVATE ${SANITIZER_FLAGS})
        target_link_options(test_integration PRIVATE ${SANITIZER_FLAGS})
        
        message(STATUS "Sanitizers enabled for tests")
    else()
        message(WARNING "Sanitizers not supported with this compiler")
    endif()
endif()

# Print test configuration summary
message(STATUS "Test configuration:")
message(STATUS "  Unit tests: test_end2end")
message(STATUS "  Integration tests: test_integration") 
message(STATUS "  Performance tests: test_performance_comparison")
if(VALGRIND_EXECUTABLE)
    message(STATUS "  Memory check: Available with Valgrind")
endif()
if(ENABLE_COVERAGE)
    message(STATUS "  Coverage: Enabled")
endif()
if(ENABLE_SANITIZERS)
    message(STATUS "  Sanitizers: Enabled")
endif() 