
cmake_minimum_required(VERSION 3.18)
project(inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(inference
    src/main.cpp
    src/safetensor.cpp
    src/tensor.cpp
    src/model.cpp
    src/storage.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inference PRIVATE
        -O3
        -march=native
    )
endif()

# Link-time optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported)

if(lto_supported)
    set_property(TARGET inference PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()


target_include_directories(inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(CTest)
if(BUILD_TESTING)
    add_executable(tensor_tests tests/tensor_tests.cpp src/tensor.cpp src/storage.cpp)
    target_include_directories(tensor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    foreach(test_case IN ITEMS
        dense_layout
        dense_arithmetic
        allocated_size
        view_size
        view_indexing
        view_iteration
        view_arithmetic
        stepped_slice
        stepped_slice_remainder
        inner_axis_slice
        multi_axis_slice
        nested_slice
        out_of_bounds_index
        zero_slice_step
        empty_tensor
        scalar_tensor
        incompatible_shapes
    )
        add_test(NAME tensor.${test_case} COMMAND tensor_tests ${test_case})
    endforeach()
endif()
